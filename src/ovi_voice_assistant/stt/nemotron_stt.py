"""Streaming speech-to-text using NVIDIA Nemotron Speech 600M ONNX.

Uses danielbodart/nemotron-speech-600m-onnx model files with direct
onnxruntime inference — no sherpa-onnx dependency.  Mel spectrogram
extraction, cache-aware streaming encoder, and RNNT greedy decoding
are all implemented in numpy + onnxruntime.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ovi_voice_assistant.config import CACHE_DIR, Settings
from ovi_voice_assistant.stt.stt import STT, VadStartCallback

logger = logging.getLogger(__name__)

# ── Silero VAD parameters (shared with WhisperSTT) ──────────────

VAD_THRESHOLD = 0.4
SILENCE_TIMEOUT_S = 0.75
MIN_SPEECH_S = 0.3
MAX_LISTEN_S = 60.0
NO_SPEECH_TIMEOUT_S = 5.0
VAD_CHUNK_SAMPLES = 512

# ── Mel spectrogram ─────────────────────────────────────────────

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 128
PREEMPH = 0.97
LOG_GUARD = 2**-24  # ~5.96e-08

# ── Encoder (cache-aware streaming FastConformer) ────────────────

MEL_SHIFT = 56  # new mel frames per encoder chunk (560 ms)
PRE_ENCODE_CACHE = 9  # mel context frames prepended to each chunk
ENC_LAYERS = 24
ENC_DIM = 1024
CACHE_CH_DIM = 70
CACHE_TIME_DIM = 8

# ── Decoder (RNNT prediction network + joint) ───────────────────

BLANK_ID = 1024
MAX_SYMBOLS_PER_FRAME = 10
PRED_HIDDEN = 640

# ── Model source ────────────────────────────────────────────────

HF_REPO = "danielbodart/nemotron-speech-600m-onnx"
VARIANTS = {"fp32", "fp16", "int8-dynamic", "int8-static"}
DEFAULT_VARIANT = "int8-dynamic"


class NemotronSTT(STT):
    """Nemotron Speech 600 M with direct ONNX Runtime inference."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._encoder: ort.InferenceSession | None = None
        self._decoder: ort.InferenceSession | None = None
        self._enc_in: list[str] = []  # encoder input tensor names
        self._dec_in: list[str] = []  # decoder input tensor names
        self._filterbank_f64: np.ndarray | None = None  # pre-cast to float64
        self._hann: np.ndarray | None = None
        self._tokens: list[str] | None = None
        self._vad_model = None

    # ── load ─────────────────────────────────────────────────────

    def load(self) -> None:
        from huggingface_hub import hf_hub_download

        variant = self._settings.stt.model
        if variant not in VARIANTS:
            variant = DEFAULT_VARIANT

        logger.info("Loading Nemotron Speech (%s)…", variant)

        cd = str(CACHE_DIR / "nemotron")
        enc_path = hf_hub_download(
            HF_REPO, f"{variant}/encoder_model.onnx", cache_dir=cd
        )
        hf_hub_download(HF_REPO, f"{variant}/encoder_model.onnx.data", cache_dir=cd)
        dec_path = hf_hub_download(
            HF_REPO, f"{variant}/decoder_model.onnx", cache_dir=cd
        )
        hf_hub_download(HF_REPO, f"{variant}/decoder_model.onnx.data", cache_dir=cd)
        fb_path = hf_hub_download(HF_REPO, "shared/filterbank.bin", cache_dir=cd)
        tok_path = hf_hub_download(HF_REPO, "shared/tokens.txt", cache_dir=cd)

        providers = self._select_providers()
        enc_opts = self._make_session_options(is_decoder=False)
        dec_opts = self._make_session_options(is_decoder=True)

        self._encoder = ort.InferenceSession(
            enc_path, sess_options=enc_opts, providers=providers
        )
        self._decoder = ort.InferenceSession(
            dec_path, sess_options=dec_opts, providers=providers
        )
        logger.info(
            "Nemotron ORT providers=%s, encoder threads=%d, decoder threads=%d",
            self._encoder.get_providers(),
            enc_opts.intra_op_num_threads,
            dec_opts.intra_op_num_threads,
        )
        self._enc_in = [i.name for i in self._encoder.get_inputs()]
        self._dec_in = [i.name for i in self._decoder.get_inputs()]

        fb = np.frombuffer(Path(fb_path).read_bytes(), dtype=np.float32).reshape(
            N_MELS, N_FFT // 2 + 1
        )
        self._filterbank_f64 = fb.astype(np.float64)  # pre-cast once

        self._tokens = []
        with open(tok_path) as f:
            for line in f:
                parts = line.strip().rsplit(" ", 1)
                self._tokens.append(parts[0] if parts else "")

        # Symmetric Hann window zero-padded to N_FFT, kept as float64
        # so the entire mel pipeline runs in float64 (matches reference).
        hann = np.zeros(N_FFT, dtype=np.float64)
        wo = (N_FFT - WIN_LENGTH) // 2
        i = np.arange(WIN_LENGTH, dtype=np.float64)
        hann[wo : wo + WIN_LENGTH] = 0.5 * (
            1.0 - np.cos(2.0 * np.pi * i / (WIN_LENGTH - 1))
        )
        self._hann = hann

        from faster_whisper.vad import get_vad_model

        self._vad_model = get_vad_model()
        logger.info("Nemotron STT ready (Silero VAD)")

    def _select_providers(self) -> list[str]:
        available = set(ort.get_available_providers())
        if self._settings.stt.device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        preferred = [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    def _make_session_options(self, is_decoder: bool) -> ort.SessionOptions:
        """Build ORT session options tuned for encoder vs decoder workloads.

        Encoder: large per-call matmuls that parallelize well → use most cores.
        Decoder: tiny per-token ops called in a tight loop → threading overhead
        outweighs gains; keep intra-op threads low.
        """
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        total = os.cpu_count() or 4
        opts.intra_op_num_threads = 2 if is_decoder else max(1, min(8, total // 2))
        return opts

    # ── mel spectrogram ──────────────────────────────────────────

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """Log-mel spectrogram [N_MELS, n_frames] in float32.

        All intermediate math is float64 for precision.
        """
        pre = np.empty(len(audio), dtype=np.float64)
        pre[0] = float(audio[0])
        pre[1:] = audio[1:].astype(np.float64) - PREEMPH * audio[:-1].astype(np.float64)

        padded = np.pad(pre, N_FFT // 2, mode="reflect")
        nf = (len(padded) - N_FFT) // HOP_LENGTH + 1
        idx = np.arange(N_FFT)[None, :] + np.arange(nf)[:, None] * HOP_LENGTH
        frames = padded[idx] * self._hann  # float64

        power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
        mel = np.dot(power, self._filterbank_f64.T)
        return np.log(mel + LOG_GUARD).astype(np.float32).T

    # ── RNNT decode ──────────────────────────────────────────────

    def _decode_chunks(
        self,
        mel: np.ndarray,
        cursor: int,
        pre_cache: np.ndarray,
        cache_ch: np.ndarray,
        cache_time: np.ndarray,
        cache_ch_len: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
        last_tok: int,
        tokens: list[int],
    ) -> tuple[
        int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int
    ]:
        """Process complete MEL_SHIFT chunks starting at *cursor*.

        Tokens are appended to *tokens* in-place.  Returns updated
        (cursor, pre_cache, cache_ch, cache_time, cache_ch_len, s1, s2, last_tok).
        """
        en, dn = self._enc_in, self._dec_in

        # Per-call decoder input buffers — local so concurrent transcribes
        # on the same STT instance don't race on shared state.
        tgt = np.zeros((1, 1), dtype=np.int32)
        tgt_len = np.array([1], dtype=np.int32)

        while cursor + MEL_SHIFT <= mel.shape[1]:
            mc = mel[:, cursor : cursor + MEL_SHIFT]
            ei = np.concatenate([pre_cache, mc], axis=1)

            eo = self._encoder.run(
                None,
                {
                    en[0]: ei[np.newaxis, :, :],
                    en[1]: np.array([ei.shape[1]], dtype=np.int64),
                    en[2]: cache_ch,
                    en[3]: cache_time,
                    en[4]: cache_ch_len,
                },
            )
            enc_out, enc_len = eo[0], int(eo[1][0])
            cache_ch, cache_time, cache_ch_len = eo[2], eo[3], eo[4]

            pre_cache = (
                mc[:, -PRE_ENCODE_CACHE:] if mc.shape[1] >= PRE_ENCODE_CACHE else mc
            )

            for t in range(enc_len):
                ef = enc_out[:, :, t : t + 1]
                for _ in range(MAX_SYMBOLS_PER_FRAME):
                    tgt[0, 0] = last_tok
                    do = self._decoder.run(
                        None,
                        {dn[0]: ef, dn[1]: tgt, dn[2]: tgt_len, dn[3]: s1, dn[4]: s2},
                    )
                    tid = int(np.argmax(do[0].flatten()))
                    if tid == BLANK_ID:
                        break  # do NOT update LSTM states on blank
                    s1, s2 = do[2], do[3]
                    tokens.append(tid)
                    last_tok = tid

            cursor += MEL_SHIFT

        return cursor, pre_cache, cache_ch, cache_time, cache_ch_len, s1, s2, last_tok

    def _tokens_to_text(self, ids: list[int]) -> str:
        if not ids or self._tokens is None:
            return ""
        return (
            "".join(self._tokens[t] for t in ids if 0 <= t < len(self._tokens))
            .replace("\u2581", " ")
            .strip()
        )

    # ── init helpers ─────────────────────────────────────────────

    @staticmethod
    def _init_caches():
        return (
            np.zeros((1, ENC_LAYERS, CACHE_CH_DIM, ENC_DIM), dtype=np.float32),
            np.zeros((1, ENC_LAYERS, ENC_DIM, CACHE_TIME_DIM), dtype=np.float32),
            np.zeros((1,), dtype=np.int64),
            np.zeros((N_MELS, PRE_ENCODE_CACHE), dtype=np.float32),
        )

    @staticmethod
    def _init_dec_states():
        return (
            np.zeros((2, 1, PRED_HIDDEN), dtype=np.float32),
            np.zeros((2, 1, PRED_HIDDEN), dtype=np.float32),
            0,  # last_tok
        )

    # ── public: batch transcribe ─────────────────────────────────

    def transcribe(self, audio_bytes: bytes) -> str:
        if self._encoder is None:
            raise RuntimeError("Call load() first")

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < self._settings.mic.sample_rate * 0.1:
            return ""

        mel = self._compute_mel(audio)
        cache_ch, cache_time, cache_ch_len, pre_cache = self._init_caches()
        s1, s2, last_tok = self._init_dec_states()
        tokens: list[int] = []

        self._decode_chunks(
            mel,
            0,
            pre_cache,
            cache_ch,
            cache_time,
            cache_ch_len,
            s1,
            s2,
            last_tok,
            tokens,
        )

        text = self._tokens_to_text(tokens)
        if text:
            logger.debug("Transcribed: %r", text)
        return text

    # ── public: streaming transcribe ─────────────────────────────

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        on_vad_start: VadStartCallback | None = None,
    ) -> str:
        """Accumulate audio with Silero VAD, run encoder chunks as mel
        frames become available, and decode tokens incrementally.
        """
        if self._encoder is None:
            raise RuntimeError("Call load() first")

        pad = N_FFT // 2
        loop = asyncio.get_running_loop()
        listen_start = loop.time()
        chunk_bytes = VAD_CHUNK_SAMPLES * 2

        # Audio / mel state
        preemph_prev: float = 0.0
        audio_chunks_buf: list[np.ndarray] = []  # list of pre-emph chunks
        audio_buf = np.zeros(0, dtype=np.float64)  # flattened on demand
        audio_len = 0
        mel_cache_parts: list[np.ndarray] = []  # list of mel slices
        mel_cache = np.zeros((N_MELS, 0), dtype=np.float32)
        mel_computed = 0

        # Encoder / decoder state
        cache_ch, cache_time, cache_ch_len, pre_cache = self._init_caches()
        s1, s2, last_tok = self._init_dec_states()
        tokens: list[int] = []
        cursor = 0  # mel frames consumed by encoder
        _last_partial = ""

        # VAD state
        vad_buf = b""
        speech_detected = False
        speech_bytes = 0
        silence_start: float | None = None
        h = np.zeros((1, 1, 128), dtype="float32")
        c = np.zeros((1, 1, 128), dtype="float32")
        vad_ctx = np.zeros((1, 64), dtype="float32")

        async for chunk in audio_chunks:
            elapsed = loop.time() - listen_start
            if elapsed > MAX_LISTEN_S:
                logger.warning("Max listen duration (%.0fs)", MAX_LISTEN_S)
                break
            if not speech_detected and elapsed > NO_SPEECH_TIMEOUT_S:
                logger.info("No speech after %.0fs", NO_SPEECH_TIMEOUT_S)
                break

            # ── Pre-emphasise and buffer ─────────────────────────
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float64) / 32768.0
            if len(samples) > 0:
                pre = np.empty(len(samples), dtype=np.float64)
                pre[0] = samples[0] - PREEMPH * preemph_prev
                if len(samples) > 1:
                    pre[1:] = samples[1:] - PREEMPH * samples[:-1]
                preemph_prev = float(samples[-1])
                audio_chunks_buf.append(pre)
                audio_len += len(pre)

            # ── VAD ──────────────────────────────────────────────
            vad_buf += chunk
            end_of_speech = False

            while len(vad_buf) >= chunk_bytes:
                vc = vad_buf[:chunk_bytes]
                vad_buf = vad_buf[chunk_bytes:]
                vs = np.frombuffer(vc, dtype=np.int16).astype(np.float32) / 32768.0
                awc = np.concatenate([vad_ctx[0], vs]).reshape(1, -1)
                vad_ctx = vs.reshape(1, -1)[..., -64:]
                output, h, c = self._vad_model.session.run(
                    None, {"input": awc, "h": h, "c": c}
                )
                prob = float(output[0])

                if prob > VAD_THRESHOLD:
                    if not speech_detected and on_vad_start is not None:
                        await on_vad_start()
                    speech_detected = True
                    speech_bytes += chunk_bytes
                    silence_start = None
                elif speech_detected:
                    if silence_start is None:
                        silence_start = loop.time()
                    elif loop.time() - silence_start >= SILENCE_TIMEOUT_S:
                        dur = speech_bytes / (
                            self._settings.mic.sample_rate
                            * self._settings.mic.sample_width
                        )
                        if dur >= MIN_SPEECH_S:
                            logger.info(
                                "End of speech (%.1fs speech, %.1fs silence)",
                                dur,
                                loop.time() - silence_start,
                            )
                            end_of_speech = True
                            break
                        else:
                            speech_detected = False
                            speech_bytes = 0
                            silence_start = None
                            preemph_prev = 0.0
                            audio_chunks_buf.clear()
                            audio_buf = np.zeros(0, dtype=np.float64)
                            audio_len = 0
                            mel_cache_parts.clear()
                            mel_cache = np.zeros((N_MELS, 0), dtype=np.float32)
                            mel_computed = 0
                            cache_ch, cache_time, cache_ch_len, pre_cache = (
                                self._init_caches()
                            )
                            s1, s2, last_tok = self._init_dec_states()
                            tokens.clear()
                            cursor = 0

            if end_of_speech:
                break

            # ── Incremental encoder (after speech detected) ──────
            if speech_detected and audio_len > pad:
                avail = (audio_len - pad) // HOP_LENGTH + 1
                new = avail - mel_computed
                if new > 0:
                    # Flatten audio buffer only when we have new mel to compute
                    if len(audio_chunks_buf) > 1:
                        audio_buf = np.concatenate(audio_chunks_buf)
                        audio_chunks_buf[:] = [audio_buf]
                    elif audio_chunks_buf:
                        audio_buf = audio_chunks_buf[0]
                    new_mel = self._compute_mel_range(audio_buf, mel_computed, new)
                    mel_cache_parts.append(new_mel)
                    mel_computed = avail

                # Flatten mel parts for decoder
                if len(mel_cache_parts) > 1:
                    mel_cache = np.concatenate(mel_cache_parts, axis=1)
                    mel_cache_parts[:] = [mel_cache]
                elif mel_cache_parts:
                    mel_cache = mel_cache_parts[0]

                (
                    cursor,
                    pre_cache,
                    cache_ch,
                    cache_time,
                    cache_ch_len,
                    s1,
                    s2,
                    last_tok,
                ) = self._decode_chunks(
                    mel_cache,
                    cursor,
                    pre_cache,
                    cache_ch,
                    cache_time,
                    cache_ch_len,
                    s1,
                    s2,
                    last_tok,
                    tokens,
                )

                # Debug: log partial transcript as tokens arrive
                partial = self._tokens_to_text(tokens)
                if partial and partial != _last_partial:
                    logger.debug("Partial: %r", partial)
                    _last_partial = partial

        # ── Final flush: process remaining mel frames ────────────
        if speech_detected and audio_len > pad:
            if len(audio_chunks_buf) > 1:
                audio_buf = np.concatenate(audio_chunks_buf)
            elif audio_chunks_buf:
                audio_buf = audio_chunks_buf[0]
            avail = (audio_len - pad) // HOP_LENGTH + 1
            new = avail - mel_computed
            if new > 0:
                mel_cache_parts.append(
                    self._compute_mel_range(audio_buf, mel_computed, new)
                )
            if len(mel_cache_parts) > 1:
                mel_cache = np.concatenate(mel_cache_parts, axis=1)
            elif mel_cache_parts:
                mel_cache = mel_cache_parts[0]
            remaining = mel_cache.shape[1] - cursor
            if 0 < remaining < MEL_SHIFT:
                mel_cache = np.concatenate(
                    [
                        mel_cache,
                        np.zeros((N_MELS, MEL_SHIFT - remaining), dtype=np.float32),
                    ],
                    axis=1,
                )
            self._decode_chunks(
                mel_cache,
                cursor,
                pre_cache,
                cache_ch,
                cache_time,
                cache_ch_len,
                s1,
                s2,
                last_tok,
                tokens,
            )

        text = self._tokens_to_text(tokens)
        if text:
            logger.debug("Transcribed: %r", text)
        return text

    # ── incremental mel for streaming ────────────────────────────

    def _compute_mel_range(
        self, audio_buf: np.ndarray, start: int, n: int
    ) -> np.ndarray:
        """Compute mel frames [start, start+n) from pre-emphasised audio.

        Left-edge frames use reflect padding; right edge must be available.
        Returns [N_MELS, n].
        """
        if n <= 0:
            return np.zeros((N_MELS, 0), dtype=np.float32)
        pad = N_FFT // 2
        centers = (start + np.arange(n)) * HOP_LENGTH
        offsets = np.arange(N_FFT)
        indices = centers[:, None] - pad + offsets[None, :]
        indices = np.where(indices < 0, -indices, indices)
        indices = np.clip(indices, 0, len(audio_buf) - 1)
        frames = audio_buf[indices] * self._hann
        power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
        mel = np.dot(power, self._filterbank_f64.T)
        return np.log(mel + LOG_GUARD).astype(np.float32).T

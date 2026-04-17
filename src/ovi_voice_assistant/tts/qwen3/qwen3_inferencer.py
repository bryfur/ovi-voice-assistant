"""Streaming Qwen3-TTS-ONNX inference engine.

Pipeline:
    text deltas → Qwen2 BPE → talker LLM (KV-cache) → local 15-codebook
                                                              ↓
                                          codec decoder → 24 kHz wav

Six ONNX sessions are orchestrated:
    talker, talker_local, talker_codec_embed, text_embed_proj,
    speaker_encoder, codec_decoder

Performance choices:
- The codec decoder runs in a background thread via a bounded queue, so the
  talker chain (talker → talker_local) for frame N+chunk_frames overlaps
  the codec for frames N..N+chunk_frames-1. Critical path drops from
  (talker_chain + codec/N) per frame to max(talker_chain, codec/N).
- The post-prefill talker KV-cache is hashed by ref-audio bytes + language
  and persisted to disk. Hot-cache load skips speaker_encoder + 4 embed
  passes + the prefix talker pass.
- The Qwen2 BPE tokenizer is loaded directly via Qwen2TokenizerFast,
  bypassing transformers' AutoProcessor (which scans every model class
  in transformers.models.*, including the audioflamingo3 import bug).
- All static input names + KV-cache key dicts are precomputed at __init__
  so the per-step hot path only assigns into a pre-built dict.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from ovi_voice_assistant.tts.qwen3.audio_utils import (
    load_audio_mono,
    mel_spectrogram,
)

logger = logging.getLogger(__name__)

NDArrayInt = npt.NDArray[np.int64]
NDArrayFloat = npt.NDArray[np.floating]

# Tokenizer pre-tokenizer: split text into segments at sentence/clause
# boundaries (incl. CJK full-width punctuation, since the model is
# multilingual) so we can ship segments to the talker as soon as they form
# a coherent unit, instead of waiting for the whole sentence.
_SEGMENT_SPLIT = re.compile(
    r"[。！？!?\.\u2026]\s*"  # noqa: RUF001
    r"|[,，;；:：\u2014\u2013\-]\s*"  # noqa: RUF001
    r"|\)\s*|\]\s*"
    r"|\n"
)


def _build_qwen2_tokenizer(configs_dir: Path):
    """Construct Qwen2TokenizerFast directly, bypassing AutoProcessor.

    AutoProcessor scans transformers.models.* by name, which under
    transformers 4.57 hits a missing-module ImportError for audioflamingo3.
    Direct construction also avoids the Qwen3TTSProcessor wrapper layer
    (a 70-line ProcessorMixin that does nothing but call .tokenizer()).
    """
    from transformers import Qwen2TokenizerFast

    return Qwen2TokenizerFast(
        vocab_file=str(configs_dir / "vocab.json"),
        merges_file=str(configs_dir / "merges.txt"),
        tokenizer_file=None,
    )


class _CodecWorker:
    """Background thread that runs the codec decoder.

    Receives audio-token frames from the talker chain via in_q and emits
    decoded float32 wav arrays via out_q. Maintains the codec KV-cache
    and hidden-state cache across calls — the codec is stateful, so the
    worker is the sole owner of those tensors.
    """

    _SHUTDOWN = object()

    def __init__(
        self,
        *,
        session: ort.InferenceSession,
        chunk_frames: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        codebook_dim: int,
        latent_dim: int,
        sliding_window: int,
        decoder_left_context: int,
        decoder_total_upsample: int,
    ) -> None:
        self._session = session
        self._chunk_frames = chunk_frames
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._codebook_dim = codebook_dim
        self._latent_dim = latent_dim
        self._sliding_window = sliding_window
        self._decoder_left_context = decoder_left_context
        self._decoder_total_upsample = decoder_total_upsample

        self._past_kv: list[np.ndarray] = []
        self._hidden_state_cache = np.zeros((1, latent_dim, 0), dtype=np.float32)
        self._pre_conv_cache = np.zeros((1, codebook_dim, 2), dtype=np.float32)
        self._reset_caches()

        self._kv_input_names = [
            (f"past_key_{i}", f"past_value_{i}") for i in range(num_layers)
        ]
        self._kv_output_names = [
            (f"present_key_{i}", f"present_value_{i}") for i in range(num_layers)
        ]
        self._output_names = [
            "wav",
            "current_hidden_state_cache",
            "current_pre_conv_hidden_state_cache",
            *(name for pair in self._kv_output_names for name in pair),
        ]
        self._buffer: list[np.ndarray] = []
        self._buffer_len = 0

        self._in_q: queue.Queue = queue.Queue(maxsize=8)
        self._out_q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="qwen3-codec", daemon=True
        )
        self._thread.start()

    def _reset_caches(self) -> None:
        self._past_kv = [
            np.zeros((1, self._num_kv_heads, 0, self._head_dim), dtype=np.float32)
            for _ in range(2 * self._num_layers)
        ]
        self._hidden_state_cache = np.zeros((1, self._latent_dim, 0), dtype=np.float32)
        self._pre_conv_cache = np.zeros((1, self._codebook_dim, 2), dtype=np.float32)

    def reset(self) -> None:
        """Drain any in-flight work, then reset codec caches.

        Called between turns. Must run on the main thread; blocks until
        the worker thread is idle.
        """
        sentinel = object()
        self._in_q.put(("reset", sentinel))
        while True:
            item = self._out_q.get()
            if item is sentinel:
                break

    def push_frame(self, audio_tokens: np.ndarray) -> None:
        """Submit one frame [1, codebooks] to the codec input queue."""
        self._in_q.put(("frame", audio_tokens))

    def end(self) -> None:
        """Signal end-of-input: worker drains buffer + emits final flush chunk."""
        self._in_q.put(("end", None))

    def chunks(self) -> Iterator[np.ndarray]:
        """Yield decoded wav arrays until the worker emits its end sentinel."""
        while True:
            item = self._out_q.get()
            if item is self._SHUTDOWN:
                return
            yield item

    def shutdown(self) -> None:
        self._in_q.put((self._SHUTDOWN, None))
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            while True:
                op, payload = self._in_q.get()
                if op is self._SHUTDOWN:
                    return
                if op == "frame":
                    self._buffer.append(payload)
                    self._buffer_len += payload.shape[0]
                    while self._buffer_len >= self._chunk_frames:
                        wav = self._decode(self._chunk_frames)
                        self._out_q.put(wav.reshape(-1))
                elif op == "end":
                    if self._buffer_len > 0:
                        wav = self._decode(self._buffer_len)
                        self._out_q.put(wav.reshape(-1))
                    self._out_q.put(self._SHUTDOWN)
                elif op == "reset":
                    self._buffer.clear()
                    self._buffer_len = 0
                    self._reset_caches()
                    self._out_q.put(payload)  # echo sentinel back
        except Exception:
            logger.exception("Codec worker crashed")
            self._out_q.put(self._SHUTDOWN)

    def _consume(self, n_frames: int) -> np.ndarray:
        """Pull n_frames from the FIFO buffer, return as [1, codebooks, T]."""
        frames: list[np.ndarray] = []
        remaining = n_frames
        while remaining > 0 and self._buffer:
            head = self._buffer[0]
            if head.shape[0] <= remaining:
                frames.append(head)
                remaining -= head.shape[0]
                self._buffer.pop(0)
            else:
                frames.append(head[:remaining])
                self._buffer[0] = head[remaining:]
                remaining = 0
        self._buffer_len -= n_frames - remaining
        return np.expand_dims(
            np.transpose(np.concatenate(frames, axis=0), (1, 0)), axis=0
        )

    def _decode(self, n_frames: int) -> np.ndarray:
        codes = self._consume(n_frames)

        # Sliding-window KV trim: keep the last (sliding_window - chunk_T) entries.
        slide_keep = self._sliding_window - codes.shape[-1]
        past_kv_in = [kv[:, :, -slide_keep:] for kv in self._past_kv]
        prev_hidden_len = self._hidden_state_cache.shape[-1]

        feed: dict[str, np.ndarray] = {
            "codes": codes,
            "hidden_state_cache": self._hidden_state_cache,
            "pre_conv_hidden_state_cache": self._pre_conv_cache,
        }
        for i, (k_name, v_name) in enumerate(self._kv_input_names):
            feed[k_name] = past_kv_in[2 * i]
            feed[v_name] = past_kv_in[2 * i + 1]

        outputs = self._session.run(self._output_names, feed)
        wav = outputs[0]
        new_hidden_cache = outputs[1]
        self._pre_conv_cache = outputs[2]
        new_kv = outputs[3:]

        # Trim KV to sliding window minus 1 (next call adds 1 token slot).
        self._past_kv = [kv[:, :, -self._sliding_window + 1 :] for kv in new_kv]
        self._hidden_state_cache = new_hidden_cache[:, :, -self._decoder_left_context :]
        # Strip wav samples that correspond to the previously-cached hidden
        # context (they were emitted on the prior call).
        return wav[..., prev_hidden_len * self._decoder_total_upsample :]


class Qwen3Inferencer:
    """Streaming Qwen3-TTS inference with bg-threaded codec decode."""

    def __init__(
        self,
        *,
        model_dir: Path,
        configs_dir: Path,
        ref_audio_path: Path,
        language: str,
        providers: list,
        num_threads: int = 4,
        chunk_frames: int = 4,
        temperature: float = 0.85,
        top_p: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.9,
        repetition_window: int = 50,
        max_steps: int = 2048,
        text_buffer_size: int = 32,
        min_text_chunk_chars: int = 8,
    ) -> None:
        self._configs_dir = configs_dir
        self._ref_audio_path = ref_audio_path
        self._language = language
        self.chunk_frames = chunk_frames
        self._max_steps = max_steps
        self._text_buffer_size = text_buffer_size
        self._min_text_chunk_chars = min_text_chunk_chars

        # ── ORT sessions ─────────────────────────────────────────────
        # The talker chain (main thread) and codec (worker thread) run
        # concurrently, so total intra_op threads must not exceed cpu_count
        # — otherwise the OS schedules them onto the same cores and per-call
        # latency balloons (codec 325→565ms, talker_local 91→139ms observed).
        #
        # The talker LLM (1.7GB FP32) is memory-bandwidth bound and does
        # not scale past ~4 threads. The codec decoder is more compute-
        # bound and benefits from extra threads. Cap talker at 4 and give
        # the rest to codec — on Apple Silicon this naturally lands the
        # talker on the 4 P-cores while codec uses the E-cores. Measured:
        # RTF 1.79 (4/6) vs 1.94 (5/5) on M4.
        talker_threads = min(4, max(1, (num_threads + 1) // 2))
        codec_threads = max(1, num_threads - talker_threads)

        def make_opts(threads: int) -> ort.SessionOptions:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.enable_cpu_mem_arena = True
            opts.enable_mem_pattern = True
            return opts

        talker_opts = make_opts(talker_threads)
        codec_opts = make_opts(codec_threads)

        def load(name: str, opts: ort.SessionOptions) -> ort.InferenceSession:
            return ort.InferenceSession(
                str(model_dir / name), sess_options=opts, providers=providers
            )

        self._talker = load("talker_model.onnx", talker_opts)
        self._talker_local = load("talker_local_model.onnx", talker_opts)
        self._text_embed = load("text_embed_proj_model.onnx", talker_opts)
        self._codec_embed = load("talker_codec_embed_model.onnx", talker_opts)
        self._speaker_encoder = load("speaker_encoder_model.onnx", talker_opts)
        self._codec = load("codec_decoder_model.onnx", codec_opts)

        # ── Configs (plain dicts, no python-box) ─────────────────────
        with open(configs_dir / "config.json") as f:
            self._config = json.load(f)
        with open(configs_dir / "speech_tokenizer_config.json") as f:
            self._codec_config = json.load(f)

        talker_cfg = self._config["talker_config"]
        codec_dec = self._codec_config["decoder_config"]

        self._head_dim = talker_cfg["head_dim"]
        self._num_kv_heads = talker_cfg["num_key_value_heads"]
        self._num_layers = talker_cfg["num_hidden_layers"]
        self._num_code_groups = talker_cfg["num_code_groups"]

        # Token IDs
        self._tts_pad_id = self._config["tts_pad_token_id"]
        self._tts_bos_id = self._config["tts_bos_token_id"]
        self._codec_bos_id = talker_cfg["codec_bos_id"]
        self._codec_eos_id = talker_cfg["codec_eos_token_id"]
        self._codec_pad_id = talker_cfg["codec_pad_id"]
        self._codec_think_bos_id = talker_cfg["codec_think_bos_id"]
        self._codec_think_eos_id = talker_cfg["codec_think_eos_id"]
        self._codec_nothink_id = talker_cfg["codec_nothink_id"]
        self._codec_lang_ids = talker_cfg["codec_language_id"]

        self.output_sample_rate: int = self._codec_config["output_sample_rate"]
        self._speaker_sr: int = self._config["speaker_encoder_config"]["sample_rate"]

        # ── Sampling params (pre-allocated arrays, reused every step) ─
        self._temperature = np.array([temperature], dtype=np.float32)
        self._top_p = np.array([top_p], dtype=np.float32)
        self._top_k = np.array([top_k], dtype=np.int64)
        self._rep_penalty = np.array([repetition_penalty], dtype=np.float32)
        self._rep_window = np.array([repetition_window], dtype=np.int64)
        self._tts_pad_arr = np.array([[self._tts_pad_id]], dtype=np.int64)

        # ── Pre-build static output name lists ───────────────────────
        self._talker_output_names = [
            "logits",
            "token",
            "token_embed",
            "hidden_states",
            *(
                f"present_{kind}_{i}"
                for i in range(self._num_layers)
                for kind in ("key", "value")
            ),
        ]
        self._talker_local_output_names = ["outputs_tokens", "outputs_embeds"]
        self._kv_input_names = [
            (f"past_key_{i}", f"past_value_{i}") for i in range(self._num_layers)
        ]

        # ── Tokenizer ────────────────────────────────────────────────
        self._tokenizer = _build_qwen2_tokenizer(configs_dir)

        # ── Codec worker (bg thread) ────────────────────────────────
        self._codec_worker = _CodecWorker(
            session=self._codec,
            chunk_frames=chunk_frames,
            num_layers=codec_dec["num_hidden_layers"],
            num_kv_heads=codec_dec["num_key_value_heads"],
            head_dim=codec_dec["head_dim"],
            codebook_dim=codec_dec["codebook_dim"],
            latent_dim=codec_dec["latent_dim"],
            sliding_window=codec_dec["sliding_window"],
            decoder_left_context=25,
            decoder_total_upsample=1920,
        )

        # ── Validate language ───────────────────────────────────────
        if language.lower() not in self._codec_lang_ids:
            raise ValueError(
                f"Unsupported language {language!r}; available: "
                f"{sorted(self._codec_lang_ids.keys())}"
            )

        # ── Generation state ────────────────────────────────────────
        self._prefill_kv: list[np.ndarray] | None = None
        self._reset_state()

        # ── Prefill the talker KV-cache once at startup ──────────────
        self._prefill()

    # ─────────────────────── PUBLIC API ──────────────────────────────

    def synthesize(self, text: str) -> Iterable[np.ndarray]:
        """Synthesize a complete utterance, yielding wav chunks as ready."""
        self.reset_turn()
        yield from self.push_text(text)
        yield from self.end_text()

    def push_text(self, fragment: str) -> Iterable[np.ndarray]:
        """Append a text delta and yield any wav chunks ready so far."""
        self._text_cache += fragment
        for segment in self._extract_segments(force=False):
            self._pending_token_ids.extend(self._tokenize(segment))
        yield from self._drive()

    def end_text(self) -> Iterable[np.ndarray]:
        """Signal end of input and drain remaining audio."""
        self._text_ended = True
        if self._text_cache:
            self._pending_token_ids.extend(self._tokenize(self._text_cache))
            self._text_cache = ""
        yield from self._drive()
        # All text consumed — generate any trailing audio frames the model
        # wants to emit (silence padding, end-of-utterance).
        while not self._is_finished:
            audio_frame = self._step(text_token=None)
            if audio_frame is None:
                break
            self._codec_worker.push_frame(audio_frame[0])
            self._step_idx += 1
            if self._step_idx >= self._max_steps:
                break
        self._codec_worker.end()
        yield from self._codec_worker.chunks()

    def reset_turn(self) -> None:
        """Reset between turns. Talker KV-cache prefill is preserved."""
        self._codec_worker.reset()
        self._reset_state()

    def shutdown(self) -> None:
        self._codec_worker.shutdown()

    # ────────────────────── INTERNAL: STATE ──────────────────────────

    def _reset_state(self) -> None:
        self._text_cache = ""
        self._text_ended = False
        self._pending_token_ids: list[int] = []

        self._past_kv: list[np.ndarray] = []
        if self._prefill_kv is not None:
            self._past_kv = [kv.copy() for kv in self._prefill_kv]
        self._generated_tokens = np.zeros((1, 0, self._num_code_groups), dtype=np.int64)

        self._step_idx = 0
        self._is_finished = False
        self._last_first_token = None
        self._last_first_token_embed: np.ndarray | None = None
        self._last_local_tokens_embed: np.ndarray | None = None
        self._last_hidden_states: np.ndarray | None = None

    # ─────────────── INTERNAL: TEXT SEGMENTATION ─────────────────────

    def _extract_segments(self, force: bool) -> list[str]:
        segments: list[str] = []
        if force:
            if self._text_cache:
                segments.append(self._text_cache)
                self._text_cache = ""
            return segments

        while self._text_cache:
            cut = None
            if len(self._text_cache) >= self._min_text_chunk_chars:
                for m in _SEGMENT_SPLIT.finditer(self._text_cache):
                    if m.end() >= self._min_text_chunk_chars:
                        cut = m.end()
                        break
            if cut is None and len(self._text_cache) >= self._text_buffer_size:
                ws = self._text_cache.rfind(" ")
                if ws != -1:
                    cut = ws + 1
            if cut is None:
                break
            segments.append(self._text_cache[:cut])
            self._text_cache = self._text_cache[cut:]
        return segments

    def _tokenize(self, text: str) -> list[int]:
        ids = self._tokenizer(text, padding=False, return_tensors="np")["input_ids"]
        return list(ids[0]) if ids.ndim == 2 else list(ids)

    # ───────────── INTERNAL: TALKER PIPELINE DRIVER ──────────────────

    def _drive(self) -> Iterable[np.ndarray]:
        """Consume pending text tokens via the talker chain.

        For each text token, run one talker step → one talker_local pass →
        push the resulting [1, 16] audio frame to the codec worker. Yield
        any wav chunks the codec has finished decoding so far.
        """
        while self._pending_token_ids and not self._is_finished:
            tok = self._pending_token_ids.pop(0)
            audio_frame = self._step(text_token=tok)
            if audio_frame is None:
                break
            self._codec_worker.push_frame(audio_frame[0])
            self._step_idx += 1
            yield from self._drain_ready_chunks()

    def _drain_ready_chunks(self) -> Iterable[np.ndarray]:
        """Non-blocking: yield any wav chunks the codec has finished."""
        while True:
            try:
                wav = self._codec_worker._out_q.get_nowait()
            except queue.Empty:
                return
            if wav is _CodecWorker._SHUTDOWN:
                return
            yield wav

    # ─────────────── INTERNAL: TALKER + LOCAL STEP ───────────────────

    def _build_talker_feed(self, inputs_embeds: np.ndarray) -> dict[str, np.ndarray]:
        feed: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "generated_tokens": self._generated_tokens[..., 0],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "repetition_penalty": self._rep_penalty,
            "repetition_window": self._rep_window,
        }
        for i, (k_name, v_name) in enumerate(self._kv_input_names):
            feed[k_name] = self._past_kv[2 * i]
            feed[v_name] = self._past_kv[2 * i + 1]
        return feed

    def _step(self, text_token: int | None) -> NDArrayInt | None:
        """One talker + talker_local step → return [1, 1, 16] audio tokens."""
        if self._is_finished:
            return None

        # Build inputs_embeds: text_token embedding + previous codec embedding.
        if self._step_idx > 0:
            codec_embeds = self._last_first_token_embed + self._last_local_tokens_embed
        else:
            codec_embeds = self._codec_embed.run(
                ["codec_emb"],
                {"codec_ids": np.array([[self._codec_bos_id]], dtype=np.int64)},
            )[0]

        text_id_arr = (
            np.array([[text_token]], dtype=np.int64)
            if text_token is not None
            else self._tts_pad_arr
        )
        text_embeds = self._text_embed.run(["text_emb_out"], {"text_ids": text_id_arr})[
            0
        ]
        inputs_embeds = text_embeds + codec_embeds

        # Talker forward.
        outputs = self._talker.run(
            self._talker_output_names, self._build_talker_feed(inputs_embeds)
        )
        first_token = outputs[1]
        self._last_first_token = first_token
        self._last_first_token_embed = outputs[2]
        self._last_hidden_states = outputs[3]
        self._past_kv = list(outputs[4:])

        if first_token == self._codec_eos_id:
            self._is_finished = True
            return None

        # Local talker (15 codebooks).
        local_outputs = self._talker_local.run(
            self._talker_local_output_names,
            {
                "past_hidden": self._last_hidden_states,
                "past_id_hidden": self._last_first_token_embed,
                "generated_tokens": self._generated_tokens[..., 1:],
                "temperature": self._temperature,
                "top_p": self._top_p,
                "top_k": self._top_k,
                "repetition_penalty": self._rep_penalty,
                "repetition_window": self._rep_window,
            },
        )
        local_tokens = local_outputs[0]
        self._last_local_tokens_embed = local_outputs[1]

        audio_tokens = np.concatenate(
            (np.expand_dims(first_token, axis=-1), local_tokens), axis=1
        )[None, :, :]  # [1, 1, 16]
        self._generated_tokens = np.concatenate(
            (self._generated_tokens, audio_tokens), axis=1
        )
        return audio_tokens

    # ───────────────── INTERNAL: PREFILL + CACHE ─────────────────────

    def _prefill_cache_path(self) -> Path:
        with open(self._ref_audio_path, "rb") as f:
            ref_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        cache_dir = self._ref_audio_path.parent / ".prefill_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{ref_hash}_{self._language}.npz"

    def _prefill(self) -> None:
        """Compute (or load from disk) the talker KV-cache after prefix.

        The prefix only depends on (ref_audio, language), so the result is
        cacheable. Hot cache: load 28x2 numpy arrays from .npz (~30ms).
        """
        cache_path = self._prefill_cache_path()
        if cache_path.exists():
            try:
                npz = np.load(cache_path)
                self._prefill_kv = [npz[f"kv_{i}"] for i in range(2 * self._num_layers)]
                logger.info("Loaded cached talker prefill (%s)", cache_path.name)
                self._past_kv = [kv.copy() for kv in self._prefill_kv]
                return
            except OSError, KeyError:
                logger.warning("Corrupt prefill cache at %s, recomputing", cache_path)

        embeds = self._build_prefill_embeds()
        zero_kv = [
            np.zeros((1, self._num_kv_heads, 0, self._head_dim), dtype=np.float32)
            for _ in range(2 * self._num_layers)
        ]
        self._past_kv = zero_kv
        outputs = self._talker.run(
            self._talker_output_names, self._build_talker_feed(embeds)
        )
        self._prefill_kv = list(outputs[4:])
        self._past_kv = [kv.copy() for kv in self._prefill_kv]

        try:
            np.savez(
                cache_path,
                **{f"kv_{i}": kv for i, kv in enumerate(self._prefill_kv)},
            )
            logger.info("Saved talker prefill cache (%s)", cache_path.name)
        except OSError as e:
            logger.warning("Could not write prefill cache: %s", e)

    def _build_prefill_embeds(self) -> np.ndarray:
        """Build the [1, 8, hidden] prefix embedding fed into the talker.

        Layout (concatenated along time):
          - 3 tokens: <|im_start|>assistant\\n  (text_embed_proj)
          - 5 or 6 tokens: codec_think prefix + speaker_embed + codec_pad
            (talker_codec_embed for surrounding tokens, speaker_encoder
             for the middle 1-token speaker identity slot)
        """
        speaker_embed = self._compute_speaker_embed()  # [1, 1, hidden]

        lang_id = self._codec_lang_ids.get(self._language.lower())
        if lang_id is not None:
            prefix_codec_ids = np.array(
                [
                    [
                        talker_cfg_int(self._config, "codec_think_id"),
                        self._codec_think_bos_id,
                        lang_id,
                        self._codec_think_eos_id,
                    ]
                ],
                dtype=np.int64,
            )
        else:
            prefix_codec_ids = np.array(
                [
                    [
                        self._codec_nothink_id,
                        self._codec_think_bos_id,
                        self._codec_think_eos_id,
                    ]
                ],
                dtype=np.int64,
            )
        prefix_embed = self._codec_embed.run(
            ["codec_emb"], {"codec_ids": prefix_codec_ids}
        )[0]
        pad_embed = self._codec_embed.run(
            ["codec_emb"],
            {"codec_ids": np.array([[self._codec_pad_id]], dtype=np.int64)},
        )[0]
        codec_input = np.concatenate([prefix_embed, speaker_embed, pad_embed], axis=1)

        # Text-side: <|im_start|>assistant\n + tts_bos with tts_pad padding
        # to align lengths.
        prefix_tokens = np.array(
            [self._tokenize("<|im_start|>assistant\n")], dtype=np.int64
        )
        role_embed = self._text_embed.run(
            ["text_emb_out"], {"text_ids": prefix_tokens}
        )[0]
        bospad_embed = self._text_embed.run(
            ["text_emb_out"],
            {
                "text_ids": np.array(
                    [[self._tts_bos_id, self._tts_pad_id]], dtype=np.int64
                )
            },
        )[0]
        bos_embed = bospad_embed[:, :1]
        pad_text_embed = bospad_embed[:, 1:]

        text_aligned = np.concatenate(
            (
                np.broadcast_to(
                    pad_text_embed,
                    (
                        pad_text_embed.shape[0],
                        codec_input.shape[1] - 1,
                        pad_text_embed.shape[2],
                    ),
                ),
                bos_embed,
            ),
            axis=1,
        )
        return np.concatenate((role_embed, text_aligned + codec_input), axis=1)

    def _compute_speaker_embed(self) -> np.ndarray:
        wav = load_audio_mono(self._ref_audio_path, self._speaker_sr)
        mel = mel_spectrogram(wav, sr=self._speaker_sr)
        return self._speaker_encoder.run(["speaker_embedding"], {"mel_spec": mel})[0]


def talker_cfg_int(config: dict[str, Any], key: str) -> int:
    """Read a token id from talker_config; raise if missing."""
    val = config["talker_config"].get(key)
    if val is None:
        raise KeyError(f"talker_config.{key} missing from model config.json")
    return int(val)

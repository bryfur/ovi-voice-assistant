"""Audio utilities for Qwen3 inference: mel spectrogram + ref audio loading.

Mel spectrogram is a pure-NumPy port of the librosa/torchaudio pipeline used
by the upstream model. Audio loading uses soundfile + audresample (already
project deps) instead of librosa to avoid pulling in numba.
"""

from __future__ import annotations

from pathlib import Path

import audresample
import numpy as np
import numpy.typing as npt
import soundfile as sf


def hz_to_mel(freq: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mels: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel_filterbank(
    *, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None = None
) -> npt.NDArray[np.float32]:
    """Slaney-normalized mel filterbank (matches librosa.filters.mel)."""
    if fmax is None:
        fmax = sr / 2.0

    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_freqs, dtype=np.float64)

    m_min = hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    m_max = hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)
    hz_pts = mel_to_hz(m_pts)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left_slope = (freqs - left) / (center - left + 1e-10)
        right_slope = (right - freqs) / (right - center + 1e-10)
        fb[i] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    enorm = 2.0 / (hz_pts[2:] - hz_pts[:-2])
    fb *= enorm[:, None]
    return fb.astype(np.float32)


def _stft_magnitude(
    y: npt.NDArray[np.float32], *, n_fft: int, hop_size: int, win_size: int
) -> npt.NDArray[np.float32]:
    if y.ndim != 2 or y.shape[0] != 1:
        raise ValueError("Expected waveform shape [1, T].")

    x = y.astype(np.float32, copy=False)
    if x.shape[1] < n_fft:
        raise ValueError("Input is too short for the requested n_fft.")

    num_frames = 1 + (x.shape[1] - n_fft) // hop_size
    frame_starts = hop_size * np.arange(num_frames, dtype=np.int64)
    frame_offsets = np.arange(n_fft, dtype=np.int64)
    frames = x[:, frame_starts[:, None] + frame_offsets[None, :]]

    window = np.hanning(win_size).astype(np.float32)
    if n_fft > win_size:
        pad_left = (n_fft - win_size) // 2
        pad_right = n_fft - win_size - pad_left
        window = np.pad(window, (pad_left, pad_right))
    elif n_fft < win_size:
        window = window[:n_fft]

    frames = frames * window[None, None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    return np.sqrt(np.real(spec) ** 2 + np.imag(spec) ** 2 + 1e-9).astype(np.float32)


def mel_spectrogram(
    y: npt.NDArray[np.float32],
    *,
    n_fft: int = 1024,
    n_mels: int = 128,
    sr: int = 24000,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: int = 0,
    fmax: int | None = 12000,
    clip_val: float = 1e-5,
) -> npt.NDArray[np.float32]:
    """Log-mel spectrogram in [B, T, n_mels] for the speaker encoder."""
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=float(fmin), fmax=fmax)
    spec = _stft_magnitude(y, n_fft=n_fft, hop_size=hop_size, win_size=win_size)
    mel = np.matmul(fb[None, :, :], np.transpose(spec, (0, 2, 1)))
    mel = np.log(np.clip(mel, a_min=clip_val, a_max=None)).astype(np.float32)
    return mel.transpose(0, 2, 1)


def load_audio_mono(path: str | Path, target_sr: int) -> npt.NDArray[np.float32]:
    """Load audio file as mono float32 at target_sr."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = audio.astype(np.float32, copy=False)
    if sr != target_sr:
        audio = audresample.resample(audio.reshape(1, -1), sr, target_sr)[0]
    return audio

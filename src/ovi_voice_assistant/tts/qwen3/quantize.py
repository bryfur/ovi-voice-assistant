"""Dynamically quantize Qwen3-TTS ONNX models to int8 for ~2x CPU speedup.

Usage:
    uv run python -m ovi_voice_assistant.tts.qwen3.quantize

Quantizes transformer-heavy models (talker, talker_local, text_embed_proj)
where MatMul/Gather dominate. Leaves codec_decoder + speaker_encoder at
FP32 — quantizing the codec adds audible noise to the final waveform; the
speaker encoder runs once per ref audio and is already small.

Output goes to <cache>/qwen3-tts/qwen3-tts_onnx_int8/. Qwen3TTS prefers
that dir when present, so a successful run is "active" with no further
config.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from ovi_voice_assistant.config import CACHE_DIR

SRC_DIR = CACHE_DIR / "qwen3-tts" / "qwen3-tts_onnx"
DST_DIR = CACHE_DIR / "qwen3-tts" / "qwen3-tts_onnx_int8"

# (filename, quantize?)
_TARGETS: list[tuple[str, bool]] = [
    ("talker_model.onnx", True),
    ("talker_local_model.onnx", True),
    ("text_embed_proj_model.onnx", True),
    ("talker_codec_embed_model.onnx", True),  # tiny but still worth ~2x size cut
    ("codec_decoder_model.onnx", False),  # audio-critical, FP32
    ("speaker_encoder_model.onnx", False),  # one-shot per ref, FP32
]


def quantize_one(src: Path, dst: Path) -> tuple[float, int, int]:
    """Quantize one model. Returns (elapsed_seconds, src_mb, dst_mb).

    - per_channel=True: per-channel weight quantization (better accuracy
      for transformer MatMuls; ~10% larger than per-tensor, negligible
      runtime cost).
    - reduce_range=True: 7-bit weights instead of 8-bit. Avoids overflow
      on CPUs without VNNI (Apple Silicon, older Intel) at a small
      accuracy cost — without it, audio output develops audible artifacts.
    """
    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    return (
        time.perf_counter() - t0,
        src.stat().st_size // (1024 * 1024),
        dst.stat().st_size // (1024 * 1024),
    )


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(
            f"FP32 models missing at {SRC_DIR}. Run `uv run ovi` once to download."
        )

    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source: {SRC_DIR}")
    print(f"Output: {DST_DIR}")
    print()

    total_t = 0.0
    total_saved_mb = 0

    for name, do_quantize in _TARGETS:
        src = SRC_DIR / name
        dst = DST_DIR / name

        if not src.exists():
            print(f"  {name}: SOURCE MISSING, skipping")
            continue

        if dst.exists():
            print(f"  {name}: already done, skipping")
            continue

        if not do_quantize:
            shutil.copy2(src, dst)
            mb = src.stat().st_size // (1024 * 1024)
            print(f"  {name}: copied as-is (FP32, {mb}MB)")
            continue

        print(f"  {name}: quantizing...", flush=True)
        elapsed, src_mb, dst_mb = quantize_one(src, dst)
        total_t += elapsed
        total_saved_mb += src_mb - dst_mb
        print(f"    done in {elapsed:.1f}s  ({src_mb}MB -> {dst_mb}MB)")

    print()
    print(f"Total quantize time: {total_t:.1f}s")
    print(f"Disk saved: {total_saved_mb}MB")
    print()
    print("Qwen3TTS will now use int8 models automatically on next load.")


if __name__ == "__main__":
    main()

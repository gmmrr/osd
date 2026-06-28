#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- NORMALIZATION ---

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

try:
    import pyloudnorm as pyln
except ImportError as exc:  # pragma: no cover - dependency guard
    pyln = None
    _PYLOUDNORM_IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import (
    STEP_1_AUDIO_SAMPLE_RATE,
    STEP_2_NORMALIZE_TARGET_DBFS,
    STEP_2_NORMALIZE_LIMIT_DB,
)


def loudness_normalize(
    y: np.ndarray,
    sample_rate: int,
    target_dbfs: float = STEP_2_NORMALIZE_TARGET_DBFS,
    limit_db: float = STEP_2_NORMALIZE_LIMIT_DB,
) -> tuple[np.ndarray, float]:
    if pyln is None:
        raise RuntimeError(
            "pyloudnorm is required for LUFS normalization. "
            "Install it with: uv pip install pyloudnorm"
        ) from _PYLOUDNORM_IMPORT_ERROR

    if y.size == 0 or np.allclose(y, 0):
        return y, 0.0

    meter = pyln.Meter(sample_rate)
    measured_lufs = float(meter.integrated_loudness(y))
    if not np.isfinite(measured_lufs):
        return y, 0.0

    gain_db = max(-limit_db, min(limit_db, target_dbfs - measured_lufs))
    factor = 10 ** (gain_db / 20)
    normalized = y * factor

    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.98:
        normalized = normalized * (0.98 / peak)

    return normalized, gain_db


def normalize_single_audio(
    input_file: Path,
    force: bool = False,
) -> Path:
    input_file = Path(input_file)
    if not input_file.name.endswith("_std.wav"):
        raise ValueError(f"Normalize expects *_std.wav input, got: {input_file.name}")

    out_path = input_file.with_name(f"{input_file.stem}_nml.wav")
    if out_path.exists() and not force:
        print(f"↪ {input_file.name}: normalized file already exists (cached)")
        return out_path

    y, sr = sf.read(str(input_file), always_2d=True, dtype="float32")
    y = y.mean(axis=1)

    if sr != STEP_1_AUDIO_SAMPLE_RATE:
        g = math.gcd(STEP_1_AUDIO_SAMPLE_RATE, sr)
        y = scipy.signal.resample_poly(y, STEP_1_AUDIO_SAMPLE_RATE // g, sr // g)
        sr = STEP_1_AUDIO_SAMPLE_RATE

    y, gain_db = loudness_normalize(
        y,
        sr,
        target_dbfs=STEP_2_NORMALIZE_TARGET_DBFS,
        limit_db=STEP_2_NORMALIZE_LIMIT_DB,
    )

    sf.write(out_path, y, sr, subtype="PCM_16")
    print(f"Normalized: {input_file.name} -> {out_path.name} ({gain_db:+.2f} LU, {sr} Hz, mono, LUFS)")
    return out_path


def run_normalize_dir(
    input_dir: Path | str,
    force: bool = False,
) -> None:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    paths = [path for path in sorted(input_dir.iterdir()) if path.is_file() and path.suffix.lower() == ".wav"]
    input_files = [path for path in paths if path.name.endswith("_std.wav")]
    invalid_files = [path.name for path in paths if not path.name.endswith("_std.wav")]
    if invalid_files:
        raise ValueError(f"Normalize expects only *_std.wav inputs, got: {', '.join(invalid_files[:5])}")
    if not input_files:
        print(f"⚠️  No standardized files found in: {input_dir}")
        return

    print(f"🚀 Normalize in '{input_dir}'")
    print(f"   • Files: {len(input_files)}")
    print(f"   • Analysis SR: {STEP_1_AUDIO_SAMPLE_RATE}")

    for file in input_files:
        normalize_single_audio(file, force=force)

    print("✅ Normalization completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run normalization over a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_std.wav files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_normalize_dir(args.input_dir, force=args.force)


if __name__ == "__main__":
    main()

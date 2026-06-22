#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- NORMALIZATION ---

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    import pyloudnorm as pyln
except ImportError as exc:  # pragma: no cover - dependency guard
    pyln = None
    _PYLOUDNORM_IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.constants import EXT_WAV, KEY_NORM, KEY_STD
from config.params import (
    AUDIO_SAMPLE_RATE,
    NORMALIZE_TARGET_DBFS,
    NORMALIZE_LIMIT_DB,
    NORMALIZE_WAV_SUBTYPE,
)


def loudness_normalize(
    y: np.ndarray,
    sample_rate: int,
    target_dbfs: float = NORMALIZE_TARGET_DBFS,
    limit_db: float = NORMALIZE_LIMIT_DB,
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
    if np.isnan(measured_lufs) or np.isinf(measured_lufs):
        return y, 0.0

    gain_db = float(np.clip(target_dbfs - measured_lufs, -limit_db, limit_db))
    factor = 10 ** (gain_db / 20)
    normalized = y * factor

    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.98:
        normalized = normalized * (0.98 / peak)

    return normalized, gain_db


def normalize_single_audio(
    input_file: Path,
    device: str = "auto",
    force: bool = False,
) -> Path:
    """
    Normalize one standardized file in place.

    Expected input:
        <stem>_std.wav
    Output:
        <stem>_std_nml.wav
    """
    input_file = Path(input_file)
    if not input_file.name.endswith(f"_{KEY_STD}{EXT_WAV}"):
        raise ValueError(f"Normalize expects *_std.wav input, got: {input_file.name}")

    out_path = input_file.with_name(f"{input_file.stem}_{KEY_NORM}{EXT_WAV}")
    if out_path.exists() and not force:
        print(f"↪ {input_file.name}: normalized file already exists (cached)")
        return out_path

    y, sr = librosa.load(input_file, sr=None, mono=True)

    if sr != AUDIO_SAMPLE_RATE:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
        sr = AUDIO_SAMPLE_RATE

    y, gain_db = loudness_normalize(
        y,
        sr,
        target_dbfs=NORMALIZE_TARGET_DBFS,
        limit_db=NORMALIZE_LIMIT_DB,
    )

    sf.write(out_path, y, sr, subtype=NORMALIZE_WAV_SUBTYPE)
    print(f"Normalized: {input_file.name} -> {out_path.name} ({gain_db:+.2f} LU, {sr} Hz, mono, LUFS)")
    return out_path


def run_normalize_dir(
    input_dir: Path | str,
    device: str = "auto",
    force: bool = False,
) -> None:
    """
    Normalize all *_std.wav files in a directory.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.name.endswith(f"_{KEY_STD}{EXT_WAV}"))
    if not input_files:
        print(f"⚠️  No standardized files found in: {input_dir}")
        return

    print(f"🚀 Normalize in '{input_dir}'")
    print(f"   • Files: {len(input_files)}")
    print(f"   • Analysis SR: {AUDIO_SAMPLE_RATE}")

    for file in input_files:
        normalize_single_audio(file, device=device, force=force)

    print("✅ Normalization completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run normalization over a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_std.wav files")
    parser.add_argument("--device", default="auto", help="Device flag (accepted for CLI symmetry)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    run_normalize_dir(args.input_dir, device=args.device, force=args.force)

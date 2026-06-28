#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- STANDARDIZATION ---

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import STEP_1_AUDIO_SAMPLE_RATE

KEY_STD = "std"
EXT_WAV = ".wav"


def _is_raw_audio(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != EXT_WAV:
        return False
    stem = path.stem.lower()
    return not stem.endswith(f"_{KEY_STD}") and not stem.endswith(f"_{KEY_STD}_nml")


def standardize_single_audio(input_file: Path, force: bool = False) -> Path:
    input_file = Path(input_file)
    out_path = input_file.with_name(f"{input_file.stem}_{KEY_STD}{EXT_WAV}")
    if out_path.exists() and not force:
        print(f"↪ {input_file.name}: standardized file already exists (cached)")
        return out_path

    y, sr_in = sf.read(str(input_file), always_2d=True, dtype="float32")
    y = y.T[:2]
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    if sr_in != STEP_1_AUDIO_SAMPLE_RATE:
        g = math.gcd(STEP_1_AUDIO_SAMPLE_RATE, sr_in)
        y = np.stack(
            [
                scipy.signal.resample_poly(y[0], STEP_1_AUDIO_SAMPLE_RATE // g, sr_in // g),
                scipy.signal.resample_poly(y[1], STEP_1_AUDIO_SAMPLE_RATE // g, sr_in // g),
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        sr_in = STEP_1_AUDIO_SAMPLE_RATE

    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y = y / peak

    sf.write(out_path, y.T, sr_in, subtype="PCM_16")
    print(f"Standardized: {input_file.name} -> {out_path.name} ({sr_in} Hz, stereo)")
    return out_path


def run_standardize_dir(input_dir: Path | str, force: bool = False) -> None:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_files = sorted(p for p in input_dir.iterdir() if _is_raw_audio(p))
    if not input_files:
        print(f"⚠️  No raw audio files found in: {input_dir}")
        return

    print(f"🚀 Standardize in '{input_dir}'")
    print(f"   • Files: {len(input_files)}")
    print(f"   • Target SR: {STEP_1_AUDIO_SAMPLE_RATE}")

    for file in input_files:
        standardize_single_audio(file, force=force)

    print("✅ Standardization completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standardization over a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw audio files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    run_standardize_dir(args.input_dir, force=args.force)

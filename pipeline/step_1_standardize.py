#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- STANDARDIZATION ---

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import AUDIO_SAMPLE_RATE
from config.params import EXT_WAV, KEY_STD

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _is_raw_audio(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        return False
    stem = path.stem.lower()
    return not stem.endswith(f"_{KEY_STD}") and not stem.endswith(f"_{KEY_STD}_nml")


def standardize_single_audio(
    input_file: Path,
    device: str = "auto",
    force: bool = False,
) -> Path:
    """
    Standardize one audio file in place.

    Output:
        <input_stem>_std.wav in the same directory as the input.
    """
    input_file = Path(input_file)
    out_path = input_file.with_name(f"{input_file.stem}_{KEY_STD}{EXT_WAV}")

    if out_path.exists() and not force:
        print(f"↪ {input_file.name}: standardized file already exists (cached)")
        return out_path

    y, sr_in = librosa.load(input_file, sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    elif y.shape[0] > 2:
        y = y[:2]

    if sr_in != AUDIO_SAMPLE_RATE:
        y = np.stack(
            [
                librosa.resample(y=y[0], orig_sr=sr_in, target_sr=AUDIO_SAMPLE_RATE),
                librosa.resample(y=y[1], orig_sr=sr_in, target_sr=AUDIO_SAMPLE_RATE),
            ],
            axis=0,
        )
        sr_out = AUDIO_SAMPLE_RATE
    else:
        sr_out = int(sr_in)

    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y = y / peak

    sf.write(out_path, y.T, sr_out, subtype="PCM_16")
    print(f"Standardized: {input_file.name} -> {out_path.name} ({sr_out} Hz, stereo)")
    return out_path


def run_standardize_dir(
    input_dir: Path | str,
    device: str = "auto",
    force: bool = False,
) -> None:
    """
    Standardize all raw audio files in a directory.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_files = sorted(p for p in input_dir.iterdir() if _is_raw_audio(p))
    if not input_files:
        print(f"⚠️  No raw audio files found in: {input_dir}")
        return

    print(f"🚀 Standardize in '{input_dir}'")
    print(f"   • Files: {len(input_files)}")
    print(f"   • Target SR: {AUDIO_SAMPLE_RATE}")

    for file in input_files:
        standardize_single_audio(file, device=device, force=force)

    print("✅ Standardization completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standardization over a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw audio files")
    parser.add_argument("--device", default="auto", help="Device flag (accepted for CLI symmetry)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    run_standardize_dir(args.input_dir, device=args.device, force=args.force)

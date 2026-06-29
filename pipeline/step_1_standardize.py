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
from pipeline.utils.progress import write_progress


def standardize_single_audio(
    input_file: Path,
    force: bool = False,
    index: int | None = None,
    total: int | None = None,
) -> Path:
    """
    Step 1: Standardize one raw WAV file.

    Converts the input into mono, resamples to the target sample rate, and
    peak-normalizes the waveform before writing a 16-bit PCM WAV.

    Args:
        input_file: Raw input audio file.
        force: Overwrite the cached standardized output when True.

    Returns:
        Path to the standardized WAV file.
    """
    input_file = Path(input_file)
    out_path = input_file.with_name(f"{input_file.stem}_std.wav")
    if out_path.exists() and not force:
        if index is not None and total is not None:
            write_progress(index, total)
        return out_path

    audio, sr_in = sf.read(str(input_file), always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if sr_in != STEP_1_AUDIO_SAMPLE_RATE:
        g = math.gcd(STEP_1_AUDIO_SAMPLE_RATE, sr_in)
        audio = scipy.signal.resample_poly(audio, STEP_1_AUDIO_SAMPLE_RATE // g, sr_in // g)
        sr_in = STEP_1_AUDIO_SAMPLE_RATE

    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio / peak

    sf.write(out_path, audio, sr_in, subtype="PCM_16")
    if index is not None and total is not None:
        write_progress(index, total)
    return out_path


def run_standardize_dir(input_dir: Path | str, force: bool = False) -> None:
    """
    Step 1: Standardize all raw WAV files in a directory.

    Args:
        input_dir: Directory containing raw audio files.
        force: Overwrite cached outputs when True.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".wav"
        and not p.stem.lower().endswith("_std")
        and not p.stem.lower().endswith("_std_nml")
    )
    if not input_files:
        print(f"⚠️  No raw audio files found in: {input_dir}")
        return

    print(f"🚀 Standardize in '{input_dir}'")
    print(f"   • Files: {len(input_files)}")
    print(f"   • Target SR: {STEP_1_AUDIO_SAMPLE_RATE}")

    total = len(input_files)
    for index, file in enumerate(input_files, start=1):
        standardize_single_audio(file, force=force, index=index, total=total)

    print()
    print("✅ Standardization completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standardization over a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw audio files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_standardize_dir(args.input_dir, force=args.force)


if __name__ == "__main__":
    main()

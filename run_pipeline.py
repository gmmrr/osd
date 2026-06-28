#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the local five-step audio pipeline.

This runner executes:
  Step 1 Standardize: standardize raw WAV files
  Step 2 Normalize: normalize standardized files
  Step 3 VAD: run VAD
  Step 4 Metadata: build mixture metadata
  Step 5 Mix: render mixes and dataset manifests
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.params import (
    STEP_3_VAD_THRESHOLD,
    STEP_3_VAD_MIN_SPEECH,
    STEP_3_VAD_MIN_SILENCE,
    STEP_3_VAD_PAD,
    STEP_4_MAX_BUILD_TRIALS,
    STEP_4_MAX_MIXTURE_DURATION,
    STEP_4_MAX_SPEAKERS,
    STEP_4_MAX_SPEAKERS_PER_FRAME,
    STEP_4_MIN_MIXTURE_DURATION,
    STEP_4_MIN_SEGMENT_DURATION,
    STEP_4_OVERLAP_RATIO,
    STEP_4_OVERLAP_RANDOM_OFFSET,
    STEP_4_SEED,
)
from pipeline.step_1_standardize import run_standardize_dir
from pipeline.step_2_normalize import run_normalize_dir
from pipeline.step_3_vad import run_vad_dir
from pipeline.step_4_metadata import run_metadata_dir
from pipeline.step_5_mix import run_mix_dir


def link_workspace_inputs(input_dir: Path, work_dir: Path, force: bool) -> None:
    """
    Link raw WAV files into a staging workspace.

    Args:
        input_dir: Directory containing raw WAV files.
        work_dir: Temporary workspace used by steps 1 to 4.
        force: Recreate existing links when True.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        target = work_dir / path.name
        if target.exists() or target.is_symlink():
            if not force:
                continue
            target.unlink()
        target.symlink_to(path.resolve())


def remove_linked_raw_wavs(work_dir: Path) -> None:
    """
    Remove the linked raw WAV files from the staging workspace.

    Args:
        work_dir: Temporary workspace used by steps 1 to 4.
    """
    for path in sorted(work_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".wav":
            stem = path.stem.lower()
            if not stem.endswith("_std") and not stem.endswith("_std_nml"):
                path.unlink()


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Run the five local pipeline steps in order.

    Args:
        args: Parsed CLI arguments.
    """
    input_dir = args.input_dir
    work_dir = args.work_dir
    mix_json = work_dir / "mix.json"
    output_dir = args.output_dir

    link_workspace_inputs(input_dir, work_dir, force=args.force)

    print(f"🚀 Run pipeline in '{input_dir}'")
    print(f"   • Workspace: {work_dir}")

    print("\n▶ Step 1")
    run_standardize_dir(input_dir=work_dir, force=args.force)

    remove_linked_raw_wavs(work_dir)

    print("\n▶ Step 2")
    run_normalize_dir(input_dir=work_dir, force=args.force)

    print("\n▶ Step 3")
    run_vad_dir(
        input_dir=work_dir,
        vad_threshold=args.vad_threshold,
        vad_min_speech=args.vad_min_speech,
        vad_min_silence=args.vad_min_silence,
        vad_pad=args.vad_pad,
        force=args.force,
    )

    print("\n▶ Step 4")
    step4_args = argparse.Namespace(
        vad_dir=work_dir,
        out=mix_json,
        n_mixtures=args.n_mixtures,
        overlap_ratio=args.overlap_ratio,
        overlap_random_offset=args.overlap_random_offset,
        min_segment_duration=args.min_segment_duration,
        min_mixture_duration=args.min_mixture_duration,
        max_mixture_duration=args.max_mixture_duration,
        seed=args.seed,
        prefix="mix",
        max_build_trials=args.max_build_trials,
        max_speakers=args.max_speakers,
        max_speakers_per_frame=args.max_speakers_per_frame,
        force=args.force,
    )
    run_metadata_dir(step4_args)

    print("\n▶ Step 5")
    step5_args = argparse.Namespace(
        input=mix_json,
        output_dir=output_dir,
        force=args.force,
    )
    run_mix_dir(step5_args)

    print("\n✅ Pipeline completed.")
    print(f"   • Mix metadata: {mix_json}")
    print(f"   • Dataset root : {output_dir}")
    print(f"   • Workspace    : {work_dir}")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the five-step pipeline runner.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run the local five-step audio pipeline.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw WAV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dataset root for Step 5 outputs.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Staging workspace for Steps 1 to 4. Defaults to <output-dir>/_workspace.")
    parser.add_argument("--n-mixtures", type=int, required=True, help="Step 4 number of mixtures to generate.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")

    parser.add_argument("--vad-threshold", type=float, default=STEP_3_VAD_THRESHOLD, help="Step 3 relative RMS threshold.")
    parser.add_argument("--vad-min-speech", type=int, default=STEP_3_VAD_MIN_SPEECH, help="Step 3 minimum speech duration in ms.")
    parser.add_argument("--vad-min-silence", type=int, default=STEP_3_VAD_MIN_SILENCE, help="Step 3 minimum silence gap in ms.")
    parser.add_argument("--vad-pad", type=int, default=STEP_3_VAD_PAD, help="Step 3 padding around segments in ms.")
    parser.add_argument("--overlap-ratio", type=float, default=STEP_4_OVERLAP_RATIO, help="Step 4 overlap ratio center.")
    parser.add_argument("--overlap-random-offset", type=float, default=STEP_4_OVERLAP_RANDOM_OFFSET, help="Step 4 random offset around the overlap ratio.")
    parser.add_argument("--min-segment-duration", type=float, default=STEP_4_MIN_SEGMENT_DURATION, help="Step 4 minimum active segment duration in ms.")
    parser.add_argument("--min-mixture-duration", type=float, default=STEP_4_MIN_MIXTURE_DURATION, help="Step 4 minimum mixture duration in ms.")
    parser.add_argument("--max-mixture-duration", type=float, default=STEP_4_MAX_MIXTURE_DURATION, help="Step 4 maximum mixture duration in ms.")
    parser.add_argument("--seed", type=int, default=STEP_4_SEED, help="Step 4 random seed.")
    parser.add_argument("--max-build-trials", type=int, default=STEP_4_MAX_BUILD_TRIALS, help="Step 4 maximum sampling attempts.")
    parser.add_argument("--max-speakers", type=int, default=STEP_4_MAX_SPEAKERS, help="Step 4 maximum speakers per mixture.")
    parser.add_argument("--max-speakers-per-frame", type=int, default=STEP_4_MAX_SPEAKERS_PER_FRAME, help="Step 4 maximum overlapping speakers per frame.")

    args = parser.parse_args()
    if args.work_dir is None:
        args.work_dir = args.output_dir / "_workspace"
    return args


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

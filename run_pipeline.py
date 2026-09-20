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
import json
from pathlib import Path
from typing import Any

from config.params import (
    STEP_3_VAD_THRESHOLD,
    STEP_3_VAD_MIN_SPEECH,
    STEP_3_VAD_MIN_SILENCE,
    STEP_3_VAD_PAD,
    STEP_4_ALT_FRAME_RESOLUTION,
    STEP_4_ALT_MAX_EVENT_DURATION,
    STEP_4_ALT_MAX_SESSION_DURATION,
    STEP_4_ALT_MIN_DURATION_RATIO_THRESHOLD,
    STEP_4_ALT_MIN_EVENT_DURATION,
    STEP_4_ALT_MIN_OVERLAP_ORDER_DURATION,
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
from pipeline.alternatives.step_4_metadata_multiple import run_metadata_dir as run_metadata_multiple_dir
from pipeline.step_5_mix import run_mix_dir


def link_workspace_inputs(input_dir: Path, work_dir: Path, force: bool) -> None:
    """
    Link raw WAV files into a staging workspace.

    Args:
        input_dir: Directory containing raw WAV files.
        work_dir: Temporary workspace used by steps 1 to 4.
        force: Recreate existing links when True.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

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
        if path.is_symlink() and path.suffix.lower() == ".wav":
            stem = path.stem.lower()
            if not stem.endswith("_std") and not stem.endswith("_std_nml"):
                path.unlink()


def load_mix_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}.")
    records = [item for item in data if isinstance(item, dict)]
    if len(records) != len(data):
        raise ValueError(f"{path} must contain only JSON objects.")
    return records


def clear_dataset_outputs(output_dir: Path) -> None:
    for rel in ("database.yml",):
        path = output_dir / rel
        if path.exists():
            path.unlink()
    for folder in ("audio", "rttm", "uem", "lists"):
        root = output_dir / folder
        if root.exists():
            for path in sorted(root.glob("*")):
                if path.is_file():
                    path.unlink()


def build_step4_args(
    *,
    vad_dir: Path,
    out: Path,
    n_mixtures: int,
    args: argparse.Namespace,
    force: bool,
) -> argparse.Namespace:
    max_build_trials = max(args.max_build_trials, n_mixtures * 2)
    return argparse.Namespace(
        vad_dir=vad_dir,
        out=out,
        n_mixtures=n_mixtures,
        overlap_ratio=args.overlap_ratio,
        overlap_random_offset=args.overlap_random_offset,
        min_segment_duration=args.min_segment_duration,
        min_mixture_duration=args.min_mixture_duration,
        max_mixture_duration=args.max_mixture_duration,
        seed=args.seed,
        prefix="mix",
        max_build_trials=max_build_trials,
        max_speakers=args.max_speakers,
        max_speakers_per_frame=args.max_speakers_per_frame,
        max_speakers_per_frame_ratio=args.max_speakers_per_frame_ratio,
        frame_resolution=args.frame_resolution,
        min_event_duration=args.min_event_duration,
        max_event_duration=args.max_event_duration,
        min_overlap_order_duration=args.min_overlap_order_duration,
        min_duration_ratio_threshold=args.min_duration_ratio_threshold,
        max_session_duration=args.max_session_duration,
        force=force,
    )


def matches_prefix(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
    return actual[: len(expected)] == expected


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
    use_multiple_metadata = (
        args.max_speakers_per_frame > 2
        or args.max_speakers_per_frame_ratio is not None
        or args.min_duration_ratio_threshold != STEP_4_ALT_MIN_DURATION_RATIO_THRESHOLD
        or args.max_session_duration != STEP_4_ALT_MAX_SESSION_DURATION
    )
    metadata_runner = run_metadata_multiple_dir if use_multiple_metadata else run_metadata_dir

    if args.n_mixtures <= 0:
        raise ValueError("--n-mixtures must be positive.")
    if args.max_speakers_per_frame < 2:
        raise ValueError("--max-speakers-per-frame must be at least 2.")
    if work_dir.resolve() == output_dir.resolve():
        raise ValueError("--work-dir must be different from --output-dir.")

    existing_records = [] if args.force else load_mix_records(mix_json)
    existing_count = len(existing_records)
    resume_enabled = 0 < existing_count < args.n_mixtures
    reuse_existing = existing_count >= args.n_mixtures

    link_workspace_inputs(input_dir, work_dir, force=args.force)

    print()
    print("════════════════════════════════════════════════════════════")
    print(f"🚀 Run pipeline in '{input_dir}'")
    print(f"   • Workspace: {work_dir}")
    print("════════════════════════════════════════════════════════════")

    if reuse_existing:
        print(f"   • Reusing existing mix.json with {existing_count} record(s)")
    elif resume_enabled:
        print(f"   • Resuming from existing mix.json with {existing_count} record(s)")
    else:
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
    if args.force:
        if mix_json.exists():
            mix_json.unlink()
        metadata_runner(
            build_step4_args(
                vad_dir=work_dir,
                out=mix_json,
                n_mixtures=args.n_mixtures,
                args=args,
                force=True,
            )
        )
    elif reuse_existing:
        if existing_count > args.n_mixtures:
            mix_json.write_text(
                json.dumps(existing_records[: args.n_mixtures], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    elif resume_enabled:
        temp_mix_json = output_dir / "_mix_resume.json"
        metadata_runner(
            build_step4_args(
                vad_dir=work_dir,
                out=temp_mix_json,
                n_mixtures=args.n_mixtures,
                args=args,
                force=True,
            )
        )
        regenerated = load_mix_records(temp_mix_json)
        if not matches_prefix(existing_records, regenerated):
            temp_mix_json.unlink(missing_ok=True)
            raise RuntimeError(
                "Existing mix.json does not match the deterministic sequence for the current seed and parameters. "
                "Use --force to regenerate from scratch."
            )
        temp_mix_json.replace(mix_json)
    else:
        if mix_json.exists():
            mix_json.unlink()
        metadata_runner(
            build_step4_args(
                vad_dir=work_dir,
                out=mix_json,
                n_mixtures=args.n_mixtures,
                args=args,
                force=args.force,
            )
        )

    print("\n▶ Step 5")
    clear_dataset_outputs(output_dir)
    step5_args = argparse.Namespace(
        input=mix_json,
        output_dir=output_dir,
        force=args.force,
    )
    run_mix_dir(step5_args)

    print()
    print("════════════════════════════════════════════════════════════")
    print("✅ Pipeline completed.")
    print(f"   • Mix metadata: {mix_json}")
    print(f"   • Dataset root : {output_dir}")
    print(f"   • Workspace    : {work_dir}")
    print("════════════════════════════════════════════════════════════")


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--max-speakers-per-frame-ratio", nargs="+", help="Alternative Step 4 duration ratios for counts 1 through max-speakers-per-frame, or natural.")
    parser.add_argument("--frame-resolution", type=float, default=STEP_4_ALT_FRAME_RESOLUTION, help="Alternative Step 4 frame resolution in ms.")
    parser.add_argument("--min-event-duration", type=float, default=STEP_4_ALT_MIN_EVENT_DURATION, help="Alternative Step 4 minimum event duration in ms.")
    parser.add_argument("--max-event-duration", type=float, default=STEP_4_ALT_MAX_EVENT_DURATION, help="Alternative Step 4 maximum event duration in ms.")
    parser.add_argument("--min-overlap-order-duration", type=float, default=STEP_4_ALT_MIN_OVERLAP_ORDER_DURATION, help="Alternative Step 4 minimum cumulative duration in ms for each non-zero active overlap order.")
    parser.add_argument("--min-duration-ratio-threshold", type=float, default=STEP_4_ALT_MIN_DURATION_RATIO_THRESHOLD, help="Alternative Step 4 minimum ratio that controls mixture-level duration coverage.")
    parser.add_argument("--max-session-duration", type=float, default=STEP_4_ALT_MAX_SESSION_DURATION, help="Alternative Step 4 hard maximum session duration in ms.")

    args = parser.parse_args()
    if args.work_dir is None:
        args.work_dir = args.output_dir / "_workspace"
    return args


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

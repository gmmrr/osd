#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_PATH = REPO_ROOT / "diarization" / "sd3" / "models" / "speaker-diarization-3.0"
MAX_SPEAKERS = 3
MAX_SPEAKERS_PER_FRAME = 2


def resolve_device(device: str) -> torch.device:
    """Resolve an explicit device or prefer Apple MPS for automatic selection."""
    if device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS requested but not available, falling back to CPU.")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def discover_input_files(path: Path) -> list[Path]:
    """Return WAV files from an input directory in stable order."""
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {path}")
    return sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".wav"
    )


def resolve_pipeline_config(path: Path) -> Path:
    """Resolve a local pipeline directory to its config file."""
    if not path.is_dir():
        raise NotADirectoryError(f"Pipeline path must be a directory: {path}")
    config = path / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {config}")
    return config


def validate_model_file(path: Path) -> Path:
    """Require the adapted segmentation model path to be a file."""
    if not path.is_file():
        raise FileNotFoundError(f"Segmentation model file not found: {path}")
    return path


def validate_native_segmentation(model: Any, description: str) -> None:
    """Reject models that do not use the native segmentation-3.0 formulation."""
    specifications = model.specifications
    if isinstance(specifications, tuple):
        raise ValueError(f"{description} must expose one segmentation specification.")

    classes = list(specifications.classes)
    if not specifications.powerset:
        raise ValueError(f"{description} is not a powerset segmentation model.")
    if len(classes) != MAX_SPEAKERS:
        raise ValueError(
            f"{description} has {len(classes)} speaker classes; expected {MAX_SPEAKERS}."
        )
    if specifications.powerset_max_classes != MAX_SPEAKERS_PER_FRAME:
        raise ValueError(
            f"{description} supports {specifications.powerset_max_classes} simultaneous "
            f"speakers; expected {MAX_SPEAKERS_PER_FRAME}."
        )


def load_pipeline(
    model_path: Path | None,
    device: torch.device,
):
    """Load a local pipeline directory and optionally replace its segmentation model."""
    from pyannote.audio import Inference, Model, Pipeline

    pipeline_config = resolve_pipeline_config(DEFAULT_PIPELINE_PATH)
    pipeline = Pipeline.from_pretrained(str(pipeline_config))
    if pipeline is None:
        raise RuntimeError(
            f"Could not load local pipeline '{DEFAULT_PIPELINE_PATH}'. Check its config.yaml "
            "and referenced model files."
        )

    validate_native_segmentation(
        pipeline._segmentation.model, "Pretrained pipeline segmentation"
    )

    if model_path is not None:
        model_path = validate_model_file(model_path)
        adapted_model = Model.from_pretrained(model_path, map_location="cpu")
        if adapted_model is None:
            raise RuntimeError(f"Could not load segmentation model: {model_path}")
        validate_native_segmentation(adapted_model, "Adapted segmentation model")

        original = pipeline._segmentation.model.specifications
        adapted = adapted_model.specifications
        if adapted.duration != original.duration:
            raise ValueError(
                "Adapted segmentation chunk duration does not match the pipeline: "
                f"{adapted.duration:g}s != {original.duration:g}s."
            )

        previous_inference = pipeline._segmentation
        pipeline._segmentation = Inference(
            adapted_model,
            duration=previous_inference.duration,
            step=previous_inference.step,
            pre_aggregation_hook=previous_inference.pre_aggregation_hook,
            skip_aggregation=previous_inference.skip_aggregation,
            skip_conversion=previous_inference.skip_conversion,
            batch_size=previous_inference.batch_size,
        )
        pipeline._frames = adapted_model.example_output.frames
        pipeline.segmentation_model = adapted_model

    try:
        pipeline.to(device)
    except RuntimeError as error:
        if device.type != "mps":
            raise
        print(f"⚠️  Could not move the complete pipeline to MPS ({error}); using CPU.")
        pipeline.to(torch.device("cpu"))
    return pipeline


def write_rttm(annotation: Any, output_path: Path) -> None:
    """Write a pyannote Annotation in RTTM format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        annotation.write_rttm(stream)


def write_json(annotation: Any, output_path: Path, uri: str) -> None:
    """Write speaker turns from a pyannote Annotation as JSON."""
    segments = [
        {
            "speaker": str(speaker),
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "duration": round(float(segment.duration), 3),
        }
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    payload = {
        "uri": uri,
        "num_speakers": len(annotation.labels()),
        "num_segments": len(segments),
        "segments": segments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def print_summary(annotation: Any) -> None:
    """Print a compact human-readable diarization summary."""
    durations: dict[str, float] = defaultdict(float)
    turns: dict[str, int] = defaultdict(int)
    total_turns = 0
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        speaker = str(speaker)
        durations[speaker] += segment.duration
        turns[speaker] += 1
        total_turns += 1

    labels = [str(label) for label in annotation.labels()]
    print(f"   • Detected speakers: {', '.join(labels) if labels else 'none'}")
    print(f"   • Speaker turns: {total_turns}")
    for label in labels:
        print(
            f"     - {label}: {turns[label]} turn(s), "
            f"{durations[label]:.3f}s speech"
        )


def diarize_file(
    audio_path: Path,
    model_path: Path | None,
    output_rttm: Path | None,
    output_json: Path | None,
    device_name: str,
    pipeline: Any | None = None,
):
    """Run baseline or NvvMix-adapted full speaker diarization."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    device = resolve_device(device_name)

    if pipeline is None:
        pipeline = load_pipeline(model_path, device)
    try:
        annotation = pipeline(str(audio_path))
    except RuntimeError as error:
        pipeline_device = getattr(pipeline, "device", device)
        if pipeline_device.type != "mps":
            raise
        print("⚠️  MPS inference failed; retrying the complete pipeline on CPU.")
        try:
            pipeline.to(torch.device("cpu"))
            annotation = pipeline(str(audio_path))
        except RuntimeError as cpu_error:
            raise RuntimeError(
                f"MPS inference failed ({error}); CPU retry also failed ({cpu_error})."
            ) from cpu_error

    print_summary(annotation)
    if output_rttm is not None:
        write_rttm(annotation, output_rttm)
    if output_json is not None:
        write_json(annotation, output_json, audio_path.stem)
    return annotation


def run_diarization(
    input_dir: Path,
    output_dir: Path,
    model_path: Path | None,
    device_name: str,
    force: bool = False,
) -> None:
    """Diarize every WAV file and write matching RTTM and JSON files."""
    input_files = discover_input_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_files:
        print(f"⚠️  No WAV files found in: {input_dir}")
        return

    device = resolve_device(device_name)
    pipeline = load_pipeline(model_path, device)
    for audio_path in input_files:
        output_rttm = output_dir / f"{audio_path.stem}.rttm"
        output_json = output_dir / f"{audio_path.stem}.json"
        if output_rttm.exists() and output_json.exists() and not force:
            print(f"   • Skipping existing outputs: {output_rttm.name}, {output_json.name}")
            continue
        diarize_file(
            audio_path=audio_path,
            model_path=model_path,
            output_rttm=output_rttm,
            output_json=output_json,
            device_name=device_name,
            pipeline=pipeline,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pyannote speaker-diarization-3.0 with its original segmentation or an NvvMix-adapted segmentation model.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory containing WAV files.")
    parser.add_argument("--model-path",type=Path,help="NvvMix-adapted segmentation model file. Omit for the baseline.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where RTTM and JSON files will be written.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing RTTM and JSON files.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Execution device. auto prefers MPS, then CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = discover_input_files(args.input_dir)
    if input_files:
        device = resolve_device(args.device)
        mode = "NvvMix-adapted segmentation" if args.model_path else "original pretrained baseline"
        print(f"🚀 Diarization in '{args.input_dir}'")
        print(f"   • Files: {len(input_files)}")
        print(f"   • Mode: {mode}")
        print(f"   • Pipeline: {DEFAULT_PIPELINE_PATH}")
        print(f"   • Device: {device}")
        if args.model_path:
            print(f"   • Model: {args.model_path}")

    run_diarization(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device_name=args.device,
        force=args.force,
    )
    if input_files:
        print("✅ Diarization completed.")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()

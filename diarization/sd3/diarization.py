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
DEFAULT_OSD_MODEL = REPO_ROOT / "osd" / "models" / "pyannote-segmentation-3.0"
DEFAULT_EMBEDDING_MODEL = REPO_ROOT / "diarization" / "sd3" / "models" / "wespeaker-voxceleb-resnet34-LM"
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


def resolve_model_file(path: Path) -> Path:
    """Resolve an original model directory or a fine-tuned pyannote checkpoint."""
    path = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    checkpoint = path / "pytorch_model.bin" if path.is_dir() else path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    return checkpoint


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
    osd_model: Path,
    embedding_model: Path,
    device: torch.device,
):
    """Build the original pyannote pipeline with two selected model weights."""
    import yaml
    from pyannote.audio import Model, Pipeline

    pipeline_config = resolve_pipeline_config(DEFAULT_PIPELINE_PATH)
    segmentation = Model.from_pretrained(resolve_model_file(osd_model), map_location="cpu")
    embedding = Model.from_pretrained(resolve_model_file(embedding_model), map_location="cpu")
    if segmentation is None or embedding is None:
        raise RuntimeError("Could not load the selected segmentation or embedding model.")
    validate_native_segmentation(segmentation, "Selected segmentation model")
    if embedding.__class__.__name__ != "WeSpeakerResNet34":
        raise ValueError("Selected embedding checkpoint must be WeSpeakerResNet34.")

    # Pass loaded model objects to pyannote. pyannote.audio 4.0.7 otherwise
    # interprets any path containing 'wespeaker' as ONNX, even for PyTorch files.
    config = yaml.safe_load(pipeline_config.read_text(encoding="utf-8"))
    config["pipeline"]["params"]["segmentation"] = segmentation
    config["pipeline"]["params"]["embedding"] = embedding
    config["pipeline"]["params"]["legacy"] = True
    pipeline = Pipeline.from_pretrained(config)
    if pipeline is None:
        raise RuntimeError(f"Could not load pipeline config: {pipeline_config}")

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
    osd_model: Path,
    embedding_model: Path,
    output_rttm: Path | None,
    output_json: Path | None,
    device_name: str,
    pipeline: Any | None = None,
):
    """Run one selected segmentation and embedding combination."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    device = resolve_device(device_name)

    if pipeline is None:
        pipeline = load_pipeline(osd_model, embedding_model, device)
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

    if output_rttm is not None:
        write_rttm(annotation, output_rttm)
    if output_json is not None:
        write_json(annotation, output_json, audio_path.stem)
    return annotation


def run_diarization(
    input_dir: Path,
    output_dir: Path,
    osd_model: Path,
    embedding_model: Path,
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
    pipeline = load_pipeline(osd_model, embedding_model, device)
    skipped = 0
    total = len(input_files)
    for index, audio_path in enumerate(input_files, start=1):
        output_rttm = output_dir / f"{audio_path.stem}.rttm"
        output_json = output_dir / f"{audio_path.stem}.json"
        if output_rttm.exists() and output_json.exists() and not force:
            skipped += 1
        else:
            diarize_file(
                audio_path=audio_path,
                osd_model=osd_model,
                embedding_model=embedding_model,
                output_rttm=output_rttm,
                output_json=output_json,
                device_name=device_name,
                pipeline=pipeline,
            )
        print(
            f"Progress: {100.0 * index / total:6.2f}% ({index}/{total}) | "
            f"Skipped: {skipped}/{total}",
            end="\r",
            flush=True,
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pyannote speaker-diarization-3.0 with selected segmentation and WeSpeaker models.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory containing WAV files.")
    parser.add_argument("--osd-model", "--model-path", type=Path, default=DEFAULT_OSD_MODEL, help="Segmentation model directory or checkpoint (default: original under osd/models/).")
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_EMBEDDING_MODEL, help="WeSpeaker model directory or checkpoint (default: original under diarization/sd3/models/).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where RTTM and JSON files will be written.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing RTTM and JSON files.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Execution device. auto prefers MPS, then CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = discover_input_files(args.input_dir)
    if input_files:
        device = resolve_device(args.device)
        print(f"🚀 Diarization in '{args.input_dir}'")
        print(f"   • Files: {len(input_files)}")
        print(f"   • Pipeline: {DEFAULT_PIPELINE_PATH}")
        print(f"   • Device: {device}")
        print(f"   • Segmentation: {resolve_model_file(args.osd_model)}")
        print(f"   • Embedding: {resolve_model_file(args.embedding_model)}")

    run_diarization(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        osd_model=args.osd_model,
        embedding_model=args.embedding_model,
        device_name=args.device,
        force=args.force,
    )
    if input_files:
        print("✅ Diarization completed.")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()

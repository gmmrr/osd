#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from pyannote.core import Segment, Timeline

REPO_ROOT = Path(__file__).resolve().parents[1]
OSDC_DIR = str(REPO_ROOT / "osdc")
if OSDC_DIR in sys.path:
    sys.path.remove(OSDC_DIR)
sys.path.insert(0, str(REPO_ROOT))

from osdc.model import CLASS_LABELS, CountScores, Inference, Model

DEFAULT_MODEL_PATH = REPO_ROOT / "osdc" / "models" / "segmentation-3.0-based-v0"


@dataclass(frozen=True)
class DetectionSummary:
    files: int
    skipped: int
    sample_rate: int | None


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def discover_input_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def load_audio(path: Path, sample_rate: int) -> dict[str, object]:
    waveform, sr = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = waveform.mean(axis=1)
    if sr != sample_rate:
        factor = gcd(sample_rate, sr)
        waveform = scipy.signal.resample_poly(waveform, sample_rate // factor, sr // factor).astype(np.float32)
    return {
        "waveform": torch.from_numpy(waveform[None, :].astype(np.float32)),
        "sample_rate": sample_rate,
    }


def load_model(model_path: Path, device: torch.device):
    model = Model.from_checkpoint(model_path, map_location=device)
    return Inference(model, device=device)


def scores_to_segments(scores: CountScores) -> list[dict[str, object]]:
    data = np.asarray(scores.data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D scores, got shape {data.shape}.")
    if data.shape[0] == 0:
        return []

    counts = data.argmax(axis=1)
    edges = np.empty(len(counts) + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = scores.duration
    edges[1:-1] = 0.5 * (scores.timestamps[:-1] + scores.timestamps[1:])
    segments: list[dict[str, object]] = []
    start_idx = 0
    for idx in range(1, len(counts) + 1):
        if idx == len(counts) or counts[idx] != counts[start_idx]:
            count = int(counts[start_idx])
            start = float(edges[start_idx])
            end = float(edges[idx])
            segments.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "count": count,
                    "label": CLASS_LABELS[count],
                    "confidence": round(float(data[start_idx:idx, count].mean()), 6),
                }
            )
            start_idx = idx
    return segments


def count_segments_to_timeline(segments: Iterable[dict[str, object]], *, threshold: int = 2) -> Timeline:
    timeline = Timeline()
    for segment in segments:
        count = int(segment["count"])
        if count < threshold:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        if end > start:
            timeline.add(Segment(start, end))
    return timeline.support()


def detect_file(
    detector,
    input_path: Path,
    output_dir: Path,
    model_path: Path,
    sample_rate: int,
    force: bool,
) -> tuple[Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"{input_path.stem}_osdc.json"
    if out_json.exists() and not force:
        return out_json, True

    audio = load_audio(input_path, sample_rate)
    scores = detector(audio)
    counts = scores_to_segments(scores)
    input_duration = round(float(audio["waveform"].shape[-1]) / sample_rate, 3)

    payload = {
        "uri": input_path.stem,
        "input": str(input_path.resolve()),
        "model_path": str(model_path.resolve()),
        "pipeline_name": "Model",
        "sample_rate": sample_rate,
        "classes": list(CLASS_LABELS),
        "input_duration": input_duration,
        "num_count_segments": len(counts),
        "counts": counts,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_json, False


def run_detection(
    input_dir: Path | str,
    output_dir: Path | str,
    model_path: Path | str,
    device: str = "auto",
    force: bool = False,
) -> DetectionSummary:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_path = Path(model_path)

    input_files = discover_input_files(input_dir)
    if not input_files:
        return DetectionSummary(files=0, skipped=0, sample_rate=None)

    detector = load_model(model_path, resolve_device(device))
    sample_rate = detector.model.sample_rate

    skipped = 0
    total = len(input_files)
    for index, file_path in enumerate(input_files, start=1):
        _, did_skip = detect_file(
            detector=detector,
            input_path=file_path,
            output_dir=output_dir,
            model_path=model_path,
            sample_rate=sample_rate,
            force=force,
        )
        skipped += int(did_skip)
        print(f"Progress: {100.0 * index / total:6.2f}% ({index}/{total}) | Skipped: {skipped}/{total}", end="\r", flush=True)
    print()
    return DetectionSummary(files=total, skipped=skipped, sample_rate=sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run native {len(CLASS_LABELS)}-class OSDC inference and output JSON.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_detection(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        force=args.force,
    )
    if summary.files == 0:
        print(f"⚠️  No WAV files found in: {args.input_dir}")
        return
    print(f"🚀 OSDC in '{args.input_dir}'")
    print(f"   • Files: {summary.files}")
    print(f"   • Sample rate: {summary.sample_rate}")
    print(f"   • Model: {args.model_path}")
    print("✅ OSDC completed.")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()

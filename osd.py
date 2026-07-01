#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = REPO_ROOT / "data/test_osdc"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_osdc/local_segmentation_3_0"
DEFAULT_MODEL_PATH = REPO_ROOT / "models/pyannote-segmentation-3.0"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_ONSET = 0.5
DEFAULT_OFFSET = 0.5
DEFAULT_MIN_DURATION_ON = 0.0
DEFAULT_MIN_DURATION_OFF = 0.0


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
    from pyannote.audio import Model, Inference

    model = Model.from_pretrained(str(model_path))
    return Inference(
        model,
        device=device,
        pre_aggregation_hook=lambda scores: scores,
    )


def smooth_mask(mask: np.ndarray, max_gap: int, min_len: int) -> np.ndarray:
    if mask.size == 0:
        return mask

    while True:
        updated = mask.copy()

        start = None
        for idx, value in enumerate(updated):
            if not value and start is None:
                start = idx
            elif value and start is not None:
                if start > 0 and idx < len(updated) and idx - start <= max_gap:
                    updated[start:idx] = True
                start = None

        start = None
        for idx, value in enumerate(updated):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                if idx - start < min_len:
                    updated[start:idx] = False
                start = None

        if np.array_equal(updated, mask):
            return updated
        mask = updated


def scores_to_segments(
    scores,
    onset: float,
    offset: float,
    min_duration_on: float,
    min_duration_off: float,
) -> list[dict[str, object]]:
    data = np.asarray(scores.data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D scores, got shape {data.shape}.")
    data = np.sort(data, axis=1)[:, -2]

    active = np.zeros_like(data, dtype=bool)
    running = False
    for i, score in enumerate(data):
        if not running and score >= onset:
            running = True
        elif running and score < offset:
            running = False
        active[i] = running

    window = scores.sliding_window
    frame_step = float(getattr(window, "step", 0.0) or getattr(window, "duration", 0.0) or 0.0)
    if frame_step <= 0:
        frame_step = 1.0 / DEFAULT_SAMPLE_RATE

    active = smooth_mask(
        active,
        max(0, int(round(min_duration_off / frame_step))),
        max(0, int(round(min_duration_on / frame_step))),
    )

    segments: list[dict[str, object]] = []
    start_idx = None
    for idx, value in enumerate(active):
        if value and start_idx is None:
            start_idx = idx
        elif not value and start_idx is not None:
            start = float(window[start_idx].start)
            end = float(window[idx - 1].end)
            if end > start:
                segments.append({"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3)})
            start_idx = None

    if start_idx is not None:
        start = float(window[start_idx].start)
        end = float(window[len(active) - 1].end)
        if end > start:
            segments.append({"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3)})

    return segments


def detect_file(
    detector,
    input_path: Path,
    output_dir: Path,
    model_path: Path,
    sample_rate: int,
    onset: float,
    offset: float,
    min_duration_on: float,
    min_duration_off: float,
    force: bool,
) -> tuple[Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"{input_path.stem}_osd.json"
    if out_json.exists() and not force:
        return out_json, True

    audio = load_audio(input_path, sample_rate)
    scores = detector(audio)
    overlaps = scores_to_segments(scores, onset, offset, min_duration_on, min_duration_off)
    overlap_duration = round(sum(item["duration"] for item in overlaps), 3)
    input_duration = round(float(audio["waveform"].shape[-1]) / sample_rate, 3)

    payload = {
        "uri": input_path.stem,
        "input": str(input_path.resolve()),
        "model_path": str(model_path.resolve()),
        "pipeline_name": "pyannote.audio.Inference",
        "sample_rate": sample_rate,
        "onset": onset,
        "offset": offset,
        "min_duration_on": min_duration_on,
        "min_duration_off": min_duration_off,
        "overlap_threshold": onset,
        "input_duration": input_duration,
        "num_overlap_segments": len(overlaps),
        "overlap_duration": overlap_duration,
        "overlap_ratio": round(overlap_duration / input_duration, 6) if input_duration > 0 else 0.0,
        "overlaps": overlaps,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_json, False


def run_detection(
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    device: str = "auto",
    onset: float = DEFAULT_ONSET,
    offset: float = DEFAULT_OFFSET,
    min_duration_on: float = DEFAULT_MIN_DURATION_ON,
    min_duration_off: float = DEFAULT_MIN_DURATION_OFF,
    force: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_path = Path(model_path)

    input_files = discover_input_files(input_dir)
    if not input_files:
        print(f"⚠️  No WAV files found in: {input_dir}")
        return

    detector = load_model(model_path, resolve_device(device))

    print(f"🚀 OSD in '{input_dir}'")
    print(f"   • Files: {len(input_files)}")
    print(f"   • Sample rate: {DEFAULT_SAMPLE_RATE}")
    print(f"   • Model: {model_path}")

    skipped = 0
    total = len(input_files)
    for index, file_path in enumerate(input_files, start=1):
        _, did_skip = detect_file(
            detector=detector,
            input_path=file_path,
            output_dir=output_dir,
            model_path=model_path,
            sample_rate=DEFAULT_SAMPLE_RATE,
            onset=onset,
            offset=offset,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
            force=force,
        )
        skipped += int(did_skip)
        print(f"Progress: {100.0 * index / total:6.2f}% ({index}/{total}) | Skipped: {skipped}/{total}", end="\r", flush=True)

    print()
    print("✅ OSD completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pyannote segmentation-3.0 overlapped speech detection and output JSON.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input directory containing WAV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where OSD JSON files will be written.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Local pyannote model directory.")
    parser.add_argument("--onset", type=float, default=DEFAULT_ONSET, help="Start overlap when score >= onset.")
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET, help="End overlap when score < offset.")
    parser.add_argument("--min-duration-on", type=float, default=DEFAULT_MIN_DURATION_ON, help="Remove predicted overlap segments shorter than this many seconds.")
    parser.add_argument("--min-duration-off", type=float, default=DEFAULT_MIN_DURATION_OFF, help="Fill non-overlap gaps shorter than this many seconds.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing OSD JSON files.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_detection(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        onset=args.onset,
        offset=args.offset,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
        force=args.force,
    )
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()

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

from osd import (
    DEFAULT_MIN_DURATION_OFF,
    DEFAULT_MIN_DURATION_ON,
    DEFAULT_MODEL_PATH,
    DEFAULT_ONSET,
    DEFAULT_OFFSET,
    DEFAULT_SAMPLE_RATE,
    discover_input_files,
    load_model,
    resolve_device,
    scores_to_segments,
)


def load_audio(path: Path, sample_rate: int) -> dict[str, object]:
    waveform, sr = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = waveform.mean(axis=1)
    original_waveform = waveform.astype(np.float32, copy=True)
    original_sample_rate = sr
    if sr != sample_rate:
        factor = gcd(sample_rate, sr)
        waveform = scipy.signal.resample_poly(waveform, sample_rate // factor, sr // factor).astype(np.float32)
    return {
        "waveform": torch.from_numpy(waveform[None, :].astype(np.float32)),
        "sample_rate": sample_rate,
        "original_waveform": original_waveform,
        "original_sample_rate": original_sample_rate,
    }


def merge_intervals(segments: list[dict[str, object]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for item in sorted(segments, key=lambda x: (float(x["start"]), float(x["end"]))):
        start = float(item["start"])
        end = float(item["end"])
        if end <= start:
            continue
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def drop_overlap_audio(
    waveform: np.ndarray,
    sample_rate: int,
    overlaps: list[dict[str, object]],
    overlap_pad_ms: float,
) -> np.ndarray:
    if waveform.size == 0 or not overlaps:
        return waveform.astype(np.float32, copy=False)

    total_samples = int(waveform.shape[0])
    padded_intervals: list[tuple[float, float]] = []
    total_duration = float(total_samples) / float(sample_rate) if sample_rate > 0 else 0.0
    overlap_pad = overlap_pad_ms / 1000.0
    for item in overlaps:
        start = max(0.0, float(item["start"]) - overlap_pad)
        end = min(total_duration, float(item["end"]) + overlap_pad)
        if end > start:
            padded_intervals.append((start, end))

    merged = merge_intervals([{"start": start, "end": end} for start, end in padded_intervals])
    keep_chunks: list[np.ndarray] = []
    cursor = 0

    for start_sec, end_sec in merged:
        start = max(0, min(total_samples, int(round(start_sec * sample_rate))))
        end = max(0, min(total_samples, int(round(end_sec * sample_rate))))
        if start > cursor:
            keep_chunks.append(waveform[cursor:start])
        cursor = max(cursor, end)

    if cursor < total_samples:
        keep_chunks.append(waveform[cursor:])

    if not keep_chunks:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(keep_chunks).astype(np.float32, copy=False)


def write_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate)


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
    overlap_pad_ms: float,
    force: bool,
) -> tuple[Path, Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"{input_path.stem}_osd.json"
    out_wav = output_dir / f"{input_path.stem}_osd.wav"
    if out_json.exists() and out_wav.exists() and not force:
        return out_json, out_wav, True

    audio = load_audio(input_path, sample_rate)
    scores = detector({"waveform": audio["waveform"], "sample_rate": audio["sample_rate"]})
    overlaps = scores_to_segments(scores, onset, offset, min_duration_on, min_duration_off)

    input_waveform = np.asarray(audio["original_waveform"], dtype=np.float32)
    input_sr = int(audio["original_sample_rate"])
    dropped_waveform = drop_overlap_audio(input_waveform, input_sr, overlaps, overlap_pad_ms)

    overlap_duration = round(sum(float(item["duration"]) for item in overlaps), 3)
    input_duration = round(float(input_waveform.shape[0]) / input_sr, 3) if input_sr > 0 else 0.0
    output_duration = round(float(dropped_waveform.shape[0]) / input_sr, 3) if input_sr > 0 else 0.0

    payload = {
        "uri": input_path.stem,
        "input": str(input_path.resolve()),
        "output_audio": str(out_wav.resolve()),
        "model_path": str(model_path.resolve()),
        "pipeline_name": "pyannote.audio.Inference",
        "sample_rate": sample_rate,
        "audio_sample_rate": input_sr,
        "onset": onset,
        "offset": offset,
        "min_duration_on": min_duration_on,
        "min_duration_off": min_duration_off,
        "overlap_threshold": onset,
        "overlap_pad_ms": overlap_pad_ms,
        "input_duration": input_duration,
        "output_duration": output_duration,
        "num_overlap_segments": len(overlaps),
        "overlap_duration": overlap_duration,
        "overlap_ratio": round(overlap_duration / input_duration, 6) if input_duration > 0 else 0.0,
        "dropped_duration": round(max(0.0, input_duration - output_duration), 3),
        "dropped_ratio": round(max(0.0, input_duration - output_duration) / input_duration, 6) if input_duration > 0 else 0.0,
        "overlaps": overlaps,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_audio(out_wav, dropped_waveform, input_sr)
    return out_json, out_wav, False


def run_detection(
    input_dir: Path | str,
    output_dir: Path | str,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    device: str = "auto",
    onset: float = DEFAULT_ONSET,
    offset: float = DEFAULT_OFFSET,
    min_duration_on: float = DEFAULT_MIN_DURATION_ON,
    min_duration_off: float = DEFAULT_MIN_DURATION_OFF,
    overlap_pad_ms: float = 500.0,
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

    print(f"🚀 OSD drop in '{input_dir}'")
    print(f"   • Files: {len(input_files)}")
    print(f"   • Sample rate: {DEFAULT_SAMPLE_RATE}")
    print(f"   • Model: {model_path}")

    skipped = 0
    total = len(input_files)
    for index, file_path in enumerate(input_files, start=1):
        _, _, did_skip = detect_file(
            detector=detector,
            input_path=file_path,
            output_dir=output_dir,
            model_path=model_path,
            sample_rate=DEFAULT_SAMPLE_RATE,
            onset=onset,
            offset=offset,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
            overlap_pad_ms=overlap_pad_ms,
            force=force,
        )
        skipped += int(did_skip)
        print(f"Progress: {100.0 * index / total:6.2f}% ({index}/{total}) | Skipped: {skipped}/{total}", end="\r", flush=True)

    print()
    print("✅ OSD drop completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pyannote overlap detection, output JSON, and drop overlap regions from audio.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory containing WAV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where OSD JSON and WAV files will be written.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Local pyannote model directory.")
    parser.add_argument("--onset", type=float, default=DEFAULT_ONSET, help="Start overlap when score >= onset.")
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET, help="End overlap when score < offset.")
    parser.add_argument("--min-duration-on", type=float, default=DEFAULT_MIN_DURATION_ON, help="Remove predicted overlap segments shorter than this many seconds.")
    parser.add_argument("--min-duration-off", type=float, default=DEFAULT_MIN_DURATION_OFF, help="Fill non-overlap gaps shorter than this many seconds.")
    parser.add_argument("--overlap-pad", type=float, default=500.0, help="Extend each detected overlap segment by this many milliseconds on both sides before dropping audio.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing OSD JSON and WAV files.")
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
        overlap_pad_ms=args.overlap_pad,
        force=args.force,
    )
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()

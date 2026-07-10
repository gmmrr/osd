#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- VAD ---

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import (
    STEP_1_AUDIO_SAMPLE_RATE,
    STEP_3_VAD_ENERGY_REL_THRESHOLD,
    STEP_3_VAD_EXPAND_DELTA,
    STEP_3_VAD_EXPAND_POST,
    STEP_3_VAD_EXPAND_PRE,
    STEP_3_VAD_PAD,
    STEP_3_VAD_SMOOTHING_WINDOW,
    STEP_3_VAD_THRESHOLD,
    STEP_3_VAD_MIN_SPEECH,
    STEP_3_VAD_MIN_SILENCE,
)
from pipeline.utils.progress import write_progress

VAD_MODEL_NAME = "Silero-VAD"
_SPEAKER_ID_CACHE: dict[str, str | None] = {}


def _load_audio(path: Path, sample_rate: int = STEP_1_AUDIO_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load one WAV file and convert it to mono float32 audio.

    Args:
        path: Input WAV path.
        sample_rate: Target sample rate for resampling.

    Returns:
        A tuple of (mono waveform, sample rate).
    """
    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    if sr != sample_rate and wav.size > 0:
        g = math.gcd(sample_rate, sr)
        wav = scipy.signal.resample_poly(wav, sample_rate // g, sr // g)
        sr = sample_rate
    return wav, sr


def discover_input_files(input_dir: Path) -> list[Path]:
    """
    Find VAD input files in a directory.

    Args:
        input_dir: Directory to scan.

    Returns:
        Sorted list of .wav files.
    """
    return [path for path in sorted(input_dir.iterdir()) if path.is_file() and path.suffix.lower() == ".wav"]


def _match_wav_reference(audio_path: Path, raw_value: Any) -> bool:
    if raw_value is None:
        return False

    value = str(raw_value).strip()
    if not value:
        return False

    path_value = Path(value)
    audio_name = audio_path.name
    audio_stem = audio_path.stem
    base_stem = audio_stem.removesuffix("_nml").removesuffix("_std")
    audio_resolved = str(audio_path.resolve())

    return any(
        [
            value == audio_name,
            value == audio_stem,
            value == base_stem,
            audio_name == path_value.name,
            audio_stem == path_value.stem,
            base_stem == path_value.stem,
            value == audio_resolved,
            value.endswith(audio_name),
            value.endswith(audio_stem),
            value.endswith(base_stem),
        ]
    )


def infer_speaker(audio_path: Path) -> str | None:
    """
    Infer speaker from nearby metadata files.
    """
    cache_key = str(audio_path.resolve())
    if cache_key in _SPEAKER_ID_CACHE:
        return _SPEAKER_ID_CACHE[cache_key]

    inferred: str | None = None
    seen: set[Path] = set()
    for parent in [audio_path.parent, *audio_path.parents]:
        for name in ("metadata.jsonl", "metadata.csv"):
            metadata_path = parent / name
            if not metadata_path.exists():
                continue

            resolved = metadata_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            if resolved.suffix.lower() == ".jsonl":
                with resolved.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if isinstance(row, dict) and (
                            _match_wav_reference(audio_path, row.get("wav_path"))
                            or _match_wav_reference(audio_path, row.get("input"))
                        ):
                            speaker = row.get("speaker")
                            if speaker not in (None, ""):
                                inferred = str(speaker)
                                break
            else:
                with resolved.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        if _match_wav_reference(audio_path, row.get("wav_path")) or _match_wav_reference(
                            audio_path, row.get("input")
                        ):
                            speaker = row.get("speaker")
                            if speaker not in (None, ""):
                                inferred = str(speaker)
                                break

        if inferred is not None:
            break

    _SPEAKER_ID_CACHE[cache_key] = inferred
    return inferred


def merge_segments(segments: list[dict[str, float]]) -> list[dict[str, float]]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda seg: (float(seg["start"]), float(seg["end"])))
    merged: list[dict[str, float]] = [dict(ordered[0])]
    for seg in ordered[1:]:
        current = merged[-1]
        start = float(seg["start"])
        end = float(seg["end"])
        if start <= float(current["end"]):
            current["end"] = max(float(current["end"]), end)
        else:
            merged.append(dict(seg))
    return merged


def load_vad_model() -> torch.nn.Module:
    vad_model = load_silero_vad()
    vad_model.eval()
    return vad_model


def vad_single_audio(
    audio_path: Path,
    vad_model: torch.nn.Module,
    vad_threshold: float,
    vad_min_speech: int,
    vad_min_silence: int,
    vad_pad: int,
    out_suffix: str | None = None,
    out_dir: Path | None = None,
    force: bool = False,
    index: int | None = None,
    total: int | None = None,
) -> Path:
    """
    Step 3: Run Silero VAD on one WAV file.
    """
    audio_path = Path(audio_path)
    suffix = out_suffix or ""
    target_dir = Path(out_dir) if out_dir is not None else audio_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{audio_path.stem}_vad{suffix}.json"

    if out_path.exists() and not force:
        if index is not None and total is not None:
            write_progress(index, total)
        return out_path

    wav_cpu, sr = _load_audio(audio_path)
    wav = torch.from_numpy(np.asarray(wav_cpu, dtype=np.float32)).unsqueeze(0)

    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sr,
        threshold=vad_threshold,
        min_speech_duration_ms=vad_min_speech,
        min_silence_duration_ms=vad_min_silence,
        speech_pad_ms=vad_pad,
    )

    energy = np.abs(wav_cpu)
    window = max(1, int(STEP_3_VAD_SMOOTHING_WINDOW))
    smooth = np.convolve(energy, np.ones(window) / window, mode="same")
    mean_e = float(np.mean(smooth)) if smooth.size else 0.0
    energy_threshold = STEP_3_VAD_ENERGY_REL_THRESHOLD * mean_e
    n_samples = len(wav_cpu)
    speaker = infer_speaker(audio_path)

    step_delta = max(1, int(STEP_3_VAD_EXPAND_DELTA * sr / 1000.0))
    expanded: list[dict[str, float]] = []
    for seg in speech_timestamps:
        start = max(0, int(round(seg["start"])))
        end = min(n_samples, int(round(seg["end"])))
        start = max(0, start - int(STEP_3_VAD_EXPAND_PRE * sr / 1000.0))
        end = min(n_samples, end + int(STEP_3_VAD_EXPAND_POST * sr / 1000.0))

        while start > 0 and smooth[start] > energy_threshold:
            start = max(0, start - step_delta)
        while end < n_samples and smooth[min(end, n_samples - 1)] > energy_threshold:
            end = min(n_samples, end + step_delta)

        expanded.append({"start": round(start / sr, 6), "end": round(end / sr, 6)})

    segments = merge_segments(expanded)
    payload = {
        "input": (
            audio_path.resolve().relative_to(REPO_ROOT).as_posix()
            if audio_path.resolve().is_relative_to(REPO_ROOT)
            else audio_path.resolve().as_posix()
        ),
        "speaker": speaker,
        "model": VAD_MODEL_NAME,
        "sampling_rate": sr,
        "device": "cpu",
        "threshold": vad_threshold,
        "pad": vad_pad,
        "min_speech": vad_min_speech,
        "min_silence": vad_min_silence,
        "energy_threshold_rel": STEP_3_VAD_ENERGY_REL_THRESHOLD,
        "smoothing_window": STEP_3_VAD_SMOOTHING_WINDOW,
        "expand_pre": STEP_3_VAD_EXPAND_PRE,
        "expand_post": STEP_3_VAD_EXPAND_POST,
        "expand_delta": STEP_3_VAD_EXPAND_DELTA,
        "segment_count": len(segments),
        "segments": segments,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if index is not None and total is not None:
        write_progress(index, total)
    return out_path


def run_vad_dir(
    input_dir: Path | str,
    vad_threshold: float = STEP_3_VAD_THRESHOLD,
    vad_min_speech: int = STEP_3_VAD_MIN_SPEECH,
    vad_min_silence: int = STEP_3_VAD_MIN_SILENCE,
    vad_pad: int = STEP_3_VAD_PAD,
    force: bool = False,
    out_suffix: str | None = None,
    out_dir: Path | None = None,
) -> None:
    """
    Step 3: Run Silero VAD on every WAV file in a directory.
    """
    input_dir = Path(input_dir)
    input_files = discover_input_files(input_dir)

    vad_model = load_vad_model()
    print(f"🚀 VAD in '{input_dir}'")
    print(f"   • Files: {len(input_files)}")

    total = len(input_files)
    for index, file in enumerate(input_files, start=1):
        vad_single_audio(
            file,
            vad_model=vad_model,
            vad_threshold=vad_threshold,
            vad_min_speech=vad_min_speech,
            vad_min_silence=vad_min_silence,
            vad_pad=vad_pad,
            out_suffix=out_suffix,
            out_dir=out_dir,
            force=force,
            index=index,
            total=total,
        )

    print()
    print("✅ VAD completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Silero-based VAD over a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing .wav files")
    parser.add_argument("--threshold", type=float, default=None, help="Silero speech probability threshold.")
    parser.add_argument("--min-speech", type=int, default=STEP_3_VAD_MIN_SPEECH, help="Minimum speech segment length (ms)")
    parser.add_argument("--min-silence", type=int, default=STEP_3_VAD_MIN_SILENCE, help="Minimum silence gap (ms)")
    parser.add_argument("--pad", type=int, default=STEP_3_VAD_PAD, help="Context padding around each segment (ms)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vad_threshold = STEP_3_VAD_THRESHOLD if args.threshold is None else args.threshold
    run_vad_dir(
        args.input_dir,
        vad_threshold=vad_threshold,
        vad_min_speech=args.min_speech,
        vad_min_silence=args.min_silence,
        vad_pad=args.pad,
        force=args.force,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- RMS VAD ---

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

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    STEP_3_VAD_RMS_FRAME,
    STEP_3_VAD_RMS_HOP,
)

VAD_MODEL_NAME = "RMS-Energy-VAD"
VAD_INPUT_DIR_HELP = "Directory containing .wav files"


_SPEAKER_ID_CACHE: dict[str, str | None] = {}


def _load_audio(path: Path, sample_rate: int = STEP_1_AUDIO_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1).astype(np.float32, copy=False)
    if sr != sample_rate and wav.size > 0:
        g = math.gcd(sample_rate, sr)
        wav = scipy.signal.resample_poly(wav, sample_rate // g, sr // g).astype(np.float32, copy=False)
        sr = sample_rate
    return wav, sr


def discover_input_files(input_dir: Path) -> list[Path]:
    input_files: list[Path] = []
    invalid_files: list[str] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        if path.name.endswith("_std_nml.wav"):
            input_files.append(path)
        else:
            invalid_files.append(path.name)
    if invalid_files:
        raise ValueError(f"VAD expects only *_std_nml.wav inputs, got: {', '.join(invalid_files[:5])}")
    return input_files


def _candidate_metadata_paths(audio_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for parent in [audio_path.parent, *audio_path.parents]:
        for name in ("metadata.jsonl", "metadata.csv"):
            candidate = parent / name
            if candidate.exists():
                candidates.append(candidate)

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


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


def infer_speaker_id(audio_path: Path) -> str | None:
    cache_key = str(audio_path.resolve())
    if cache_key in _SPEAKER_ID_CACHE:
        return _SPEAKER_ID_CACHE[cache_key]

    inferred: str | None = None
    for metadata_path in _candidate_metadata_paths(audio_path):
        try:
            if metadata_path.suffix.lower() == ".jsonl":
                with metadata_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(row, dict):
                            continue
                        if _match_wav_reference(audio_path, row.get("wav_path")) or _match_wav_reference(
                            audio_path, row.get("input")
                        ):
                            speaker = row.get("speaker_id")
                            if speaker not in (None, ""):
                                inferred = str(speaker)
                                break
            else:
                with metadata_path.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        if _match_wav_reference(audio_path, row.get("wav_path")) or _match_wav_reference(
                            audio_path, row.get("input")
                        ):
                            speaker = row.get("speaker_id")
                            if speaker not in (None, ""):
                                inferred = str(speaker)
                                break
        except Exception:
            continue

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


def _mask_to_segments(
    active: np.ndarray,
    hop_length: int,
    frame_length: int,
    sample_rate: int,
    min_speech: int,
    min_silence: int,
    pad: int,
    n_samples: int,
) -> list[dict[str, float]]:
    segments: list[tuple[int, int]] = []
    start_frame: int | None = None

    for idx, is_active in enumerate(active):
        if is_active and start_frame is None:
            start_frame = idx
        elif not is_active and start_frame is not None:
            start_sample = start_frame * hop_length
            end_sample = idx * hop_length + frame_length
            segments.append((start_sample, min(end_sample, n_samples)))
            start_frame = None

    if start_frame is not None:
        start_sample = start_frame * hop_length
        segments.append((start_sample, n_samples))

    min_speech_samples = int(sample_rate * min_speech / 1000.0)
    min_silence_samples = int(sample_rate * min_silence / 1000.0)
    pad_samples = int(sample_rate * pad / 1000.0)

    filtered = [(start, end) for start, end in segments if end - start >= min_speech_samples]
    if not filtered:
        return []

    merged: list[tuple[int, int]] = [filtered[0]]
    for start, end in filtered[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= min_silence_samples:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    padded = [
        {
            "start": round(max(0, start - pad_samples) / sample_rate, 6),
            "end": round(min(n_samples, end + pad_samples) / sample_rate, 6),
        }
        for start, end in merged
    ]
    return merge_segments(padded)


def rms_vad_segments(
    wav_cpu: np.ndarray,
    sr: int,
    threshold: float,
    min_speech: int,
    min_silence: int,
    pad: int,
) -> list[dict[str, float]]:
    frame_length = max(1, int(sr * STEP_3_VAD_RMS_FRAME / 1000.0))
    hop_length = max(1, int(sr * STEP_3_VAD_RMS_HOP / 1000.0))
    if wav_cpu.size == 0:
        return []

    pad = frame_length // 2
    padded = np.pad(wav_cpu, (pad, pad), mode="constant")
    rms = np.array(
        [float(np.sqrt(np.mean(padded[start : start + frame_length] ** 2))) for start in range(0, padded.size - frame_length + 1, hop_length)],
        dtype=np.float32,
    )

    if rms.size == 0:
        return []

    max_rms = float(np.max(rms))
    if max_rms <= 0:
        return []

    active = (rms / max_rms) >= threshold
    return _mask_to_segments(
        active=active,
        hop_length=hop_length,
        frame_length=frame_length,
        sample_rate=sr,
        min_speech=min_speech,
        min_silence=min_silence,
        pad=pad,
        n_samples=len(wav_cpu),
    )


def vad_single_audio(
    audio_path: Path,
    vad_threshold: float,
    vad_min_speech: int,
    vad_min_silence: int,
    vad_pad: int,
    out_suffix: str | None = None,
    out_dir: Path | None = None,
    quiet: bool = True,
    force: bool = False,
) -> Path:
    audio_path = Path(audio_path)
    suffix = out_suffix or ""
    target_dir = Path(out_dir) if out_dir is not None else audio_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{audio_path.stem}_vad{suffix}.json"
    if out_path.exists() and not force:
        if not quiet:
            print(f"↪ {audio_path.name}: VAD json already exists (cached)")
        return out_path

    wav_cpu, sr = _load_audio(audio_path)
    segments = rms_vad_segments(
        wav_cpu=wav_cpu,
        sr=sr,
        threshold=vad_threshold,
        min_speech=vad_min_speech,
        min_silence=vad_min_silence,
        pad=vad_pad,
    )

    energy = np.abs(wav_cpu)
    smooth = np.convolve(
        energy,
        np.ones(STEP_3_VAD_SMOOTHING_WINDOW) / STEP_3_VAD_SMOOTHING_WINDOW,
        mode="same",
    )
    mean_e = float(np.mean(smooth)) if smooth.size else 0.0
    energy_threshold = STEP_3_VAD_ENERGY_REL_THRESHOLD * mean_e
    n_samples = len(wav_cpu)
    speaker_id = infer_speaker_id(audio_path)

    expanded: list[dict[str, float]] = []
    for seg in segments:
        start = max(0, int(round(float(seg["start"]) * sr)) - int(STEP_3_VAD_EXPAND_PRE * sr / 1000.0))
        end = min(n_samples, int(round(float(seg["end"]) * sr)) + int(STEP_3_VAD_EXPAND_POST * sr / 1000.0))
        step_delta = max(1, int(STEP_3_VAD_EXPAND_DELTA * sr / 1000.0))

        while start > 0 and smooth[start] > energy_threshold:
            start = max(0, start - step_delta)
        while end < n_samples - 1 and smooth[end] > energy_threshold:
            end = min(n_samples - 1, end + step_delta)

        expanded.append({"start": round(start / sr, 6), "end": round(end / sr, 6)})

    merged_segments = merge_segments(expanded)
    payload = {
        "input": str(audio_path.resolve()),
        "audio_derivative": audio_path.stem,
        "speaker_id": speaker_id,
        "sampling_rate": sr,
        "segment_count": len(merged_segments),
        "segments": merged_segments,
        "parameters": {
            "model": VAD_MODEL_NAME,
            "threshold": vad_threshold,
            "threshold_type": "relative_rms_to_max",
            "pad": vad_pad,
            "min_speech": vad_min_speech,
            "min_silence": vad_min_silence,
            "rms_frame": STEP_3_VAD_RMS_FRAME,
            "rms_hop": STEP_3_VAD_RMS_HOP,
            "energy_threshold_rel": STEP_3_VAD_ENERGY_REL_THRESHOLD,
            "smoothing_window": STEP_3_VAD_SMOOTHING_WINDOW,
            "expand_pre": STEP_3_VAD_EXPAND_PRE,
            "expand_post": STEP_3_VAD_EXPAND_POST,
            "expand_delta": STEP_3_VAD_EXPAND_DELTA,
            "device": "cpu",
            "speaker_id": speaker_id,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if not quiet:
        print(f"VAD: {audio_path.name} -> {out_path.name}")
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
    quiet: bool = True,
) -> None:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_files = discover_input_files(input_dir)
    if not input_files:
        print(f"⚠️  No .wav files found in: {input_dir}")
        return

    if not quiet:
        print(f"🚀 RMS VAD in '{input_dir}'")
        print(f"   • Device: cpu")
        print(f"   • Files: {len(input_files)}")

    for file in input_files:
        vad_single_audio(
            file,
            vad_threshold=vad_threshold,
            vad_min_speech=vad_min_speech,
            vad_min_silence=vad_min_silence,
            vad_pad=vad_pad,
            out_suffix=out_suffix,
            out_dir=out_dir,
            quiet=quiet,
            force=force,
        )

    if not quiet:
        print("✅ RMS VAD completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RMS-based VAD over a directory.")
    parser.add_argument("--input-dir", required=True, help=VAD_INPUT_DIR_HELP)
    parser.add_argument("--threshold", type=float, default=None, help="Relative RMS threshold.")
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

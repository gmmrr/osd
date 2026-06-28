#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pyannote segmentation-3.0 overlapped speech detection pipeline.

This script loads a local copy of ``pyannote/segmentation-3.0`` with
``pyannote.audio.Inference`` and writes JSON files describing regions where
two or more speakers are active.

Default demo input:
  - data/test_osdc/IyLqUS7hRvo_std_vocals.wav
"""

from __future__ import annotations

import argparse
import json
from functools import wraps
from math import gcd
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torchaudio


if not hasattr(torchaudio, "AudioMetaData"):
    class _AudioMetaData:
        """Compatibility shim for pyannote.audio import-time type hints."""

        def __init__(
            self,
            num_frames: int,
            sample_rate: int,
            num_channels: int = 1,
            bits_per_sample: int = 16,
            encoding: str = "PCM_S",
        ) -> None:
            self.num_frames = num_frames
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    torchaudio.AudioMetaData = _AudioMetaData  # type: ignore[attr-defined]

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

if not hasattr(torchaudio, "info"):
    def _info(audio_file, backend=None):
        del backend
        info = sf.info(str(audio_file))
        return torchaudio.AudioMetaData(  # type: ignore[attr-defined]
            num_frames=info.frames,
            sample_rate=info.samplerate,
            num_channels=info.channels,
            bits_per_sample=16,
            encoding=getattr(info, "subtype", "PCM_S"),
        )

    torchaudio.info = _info  # type: ignore[attr-defined]


def _allow_pyannote_checkpoint_globals() -> None:
    """Allow safe loading of pyannote checkpoints under PyTorch 2.6+."""
    try:
        from torch.serialization import add_safe_globals
        from torch.torch_version import TorchVersion
        from pyannote.audio.core.task import Problem, Resolution, Specifications

        add_safe_globals([TorchVersion, Problem, Resolution, Specifications])
    except Exception:
        pass


def _patch_torch_load_weights_only() -> None:
    """Force legacy checkpoint loads to use weights_only=False."""
    try:
        original_load = torch.load

        @wraps(original_load)
        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load  # type: ignore[assignment]
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = REPO_ROOT / "data/test_osdc/IyLqUS7hRvo_std_vocals.wav"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/test_osdc/local_segmentation_3_0"
DEFAULT_MODEL_PATH = REPO_ROOT / "models/pyannote-segmentation-3.0"
DEFAULT_PIPELINE_NAME = "pyannote.audio.Inference"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_ONSET = 0.2
DEFAULT_OFFSET = 0.1
DEFAULT_MIN_DURATION_ON = 0.0
DEFAULT_MIN_DURATION_OFF = 0.0


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU.")
        return "cpu"
    return device


def _collect_audio_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() != ".wav":
            raise ValueError(f"Expected a .wav file, got: {path}")
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
        if not files:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return files
    raise FileNotFoundError(
        f"Input path does not exist: {path}. "
        "If you are using the default input, check that the file exists under "
        "data/test_osdc/ at the repository root."
    )


def _load_audio_in_memory(audio_path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Dict[str, object]:
    waveform, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    waveform = waveform.T

    if sr != sample_rate:
        divisor = gcd(sample_rate, sr)
        up = sample_rate // divisor
        down = sr // divisor
        resampled = []
        for channel in waveform:
            resampled.append(scipy.signal.resample_poly(channel, up, down).astype(np.float32))
        max_len = max((len(channel) for channel in resampled), default=0)
        waveform = np.stack(
            [np.pad(channel, (0, max_len - len(channel)), mode="constant") for channel in resampled],
            axis=0,
        )

    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0, keepdims=True).astype(np.float32)

    return {
        "waveform": torch.from_numpy(np.asarray(waveform, dtype=np.float32)),
        "sample_rate": sample_rate,
    }


def _load_model(model_path: Path, device: str):
    _allow_pyannote_checkpoint_globals()
    original_load = torch.load
    _patch_torch_load_weights_only()

    from pyannote.audio import Model

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Local model directory not found: {model_path}")

        model = Model.from_pretrained(str(model_path))
    finally:
        torch.load = original_load  # type: ignore[assignment]

    if model is None:
        raise RuntimeError(
            f"Failed to load local pyannote model: {model_path}."
        )

    model.to(torch.device(device))
    model.eval()
    return model


def _load_detector(model_path: Path, device: str):
    from pyannote.audio import Inference

    model = _load_model(model_path=model_path, device=device)
    return Inference(model, pre_aggregation_hook=lambda scores: scores)


def _to_overlap_score(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2 or data.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)
    if data.shape[1] < 2:
        return np.zeros(data.shape[0], dtype=np.float32)
    sorted_scores = np.sort(np.asarray(data, dtype=np.float32), axis=1)
    return sorted_scores[:, -2]


def _fill_short_inactive_gaps(active: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0 or active.size == 0:
        return active

    filled = active.copy()
    idx = 0
    size = len(filled)
    while idx < size:
        if filled[idx]:
            idx += 1
            continue
        start = idx
        while idx < size and not filled[idx]:
            idx += 1
        end = idx
        gap = end - start
        has_left = start > 0 and filled[start - 1]
        has_right = end < size and filled[end]
        if has_left and has_right and gap <= max_gap_frames:
            filled[start:end] = True
    return filled


def _remove_short_active_runs(active: np.ndarray, min_frames: int) -> np.ndarray:
    if min_frames <= 1 or active.size == 0:
        return active

    cleaned = active.copy()
    idx = 0
    size = len(cleaned)
    while idx < size:
        if not cleaned[idx]:
            idx += 1
            continue
        start = idx
        while idx < size and cleaned[idx]:
            idx += 1
        end = idx
        if end - start < min_frames:
            cleaned[start:end] = False
    return cleaned


def _scores_to_overlap_segments(
    scores,
    onset: float = DEFAULT_ONSET,
    offset: float = DEFAULT_OFFSET,
    min_duration_on: float = DEFAULT_MIN_DURATION_ON,
    min_duration_off: float = DEFAULT_MIN_DURATION_OFF,
) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    if not hasattr(scores, "data") or scores.data.ndim != 2:
        return segments

    data = np.asarray(scores.data)
    window = scores.sliding_window
    overlap_score = _to_overlap_score(data)
    if overlap_score.size == 0:
        return segments

    active = np.zeros_like(overlap_score, dtype=bool)
    is_active = False
    for idx, score in enumerate(overlap_score):
        if not is_active and score >= onset:
            is_active = True
        elif is_active and score < offset:
            is_active = False
        active[idx] = is_active

    step = float(getattr(window, "step", 0.0) or 0.0)
    duration = float(getattr(window, "duration", 0.0) or 0.0)
    frame_seconds = step if step > 0 else duration
    if frame_seconds <= 0:
        frame_seconds = 1.0 / DEFAULT_SAMPLE_RATE

    off_frames = int(round(min_duration_off / frame_seconds))
    on_frames = max(1, int(round(min_duration_on / frame_seconds)))
    active = _fill_short_inactive_gaps(active, off_frames)
    active = _remove_short_active_runs(active, on_frames)

    start_idx = None
    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            start = float(window[start_idx].start)
            end = float(window[idx - 1].end)
            if end > start:
                segments.append(
                    {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "duration": round(end - start, 3),
                    }
                )
            start_idx = None

    if start_idx is not None:
        start = float(window[start_idx].start)
        end = float(window[len(active) - 1].end)
        if end > start:
            segments.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                }
            )

    return segments


def detect_single(
    detector,
    input_path: Path,
    output_dir: Path,
    model_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    onset: float = DEFAULT_ONSET,
    offset: float = DEFAULT_OFFSET,
    min_duration_on: float = DEFAULT_MIN_DURATION_ON,
    min_duration_off: float = DEFAULT_MIN_DURATION_OFF,
    force: bool = False,
) -> tuple[Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / f"{input_path.stem}_osd.json"
    if out_json.exists():
        if not force:
            return out_json, True

    audio = _load_audio_in_memory(input_path, sample_rate=sample_rate)
    result = detector(audio)

    overlaps = _scores_to_overlap_segments(
        result,
        onset=onset,
        offset=offset,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    overlap_duration = round(sum(item["duration"] for item in overlaps), 3)
    input_duration = round(float(audio["waveform"].shape[-1]) / sample_rate, 3)

    payload = {
        "uri": input_path.stem,
        "input": str(input_path.resolve()),
        "model_path": str(model_path.resolve()),
        "pipeline_name": DEFAULT_PIPELINE_NAME,
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
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    device: str = "auto",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    onset: float = DEFAULT_ONSET,
    offset: float = DEFAULT_OFFSET,
    min_duration_on: float = DEFAULT_MIN_DURATION_ON,
    min_duration_off: float = DEFAULT_MIN_DURATION_OFF,
    force: bool = False,
) -> tuple[int, int]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    model_path = Path(model_path)

    if offset > onset:
        raise ValueError(f"offset must be <= onset, got offset={offset}, onset={onset}")

    used_device = _resolve_device(device)
    detector = _load_detector(model_path=model_path, device=used_device)

    input_files = _collect_audio_files(input_path)
    skipped = 0
    total = len(input_files)
    for index, in_file in enumerate(input_files, start=1):
        out_json, did_skip = detect_single(
            detector=detector,
            input_path=in_file,
            output_dir=output_dir,
            model_path=model_path,
            sample_rate=sample_rate,
            onset=onset,
            offset=offset,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
            force=force,
        )
        skipped += int(did_skip)
        progress = 100.0 if total == 0 else 100.0 * index / total
        print(f"Progress: {progress:6.2f}% ({index}/{total}) | Skipped: {skipped}/{total}", end="\r", flush=True)

    if total > 0:
        print()
    return skipped, total


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pyannote segmentation-3.0 overlapped speech detection and output JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input audio file or directory. Default: data/test_osdc/IyLqUS7hRvo_std_vocals.wav",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where OSDC JSON files will be written.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local pyannote model directory.",
    )
    parser.add_argument("--onset", type=float, default=DEFAULT_ONSET, help="Start overlap when score >= onset.")
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET, help="End overlap when score < offset.")
    parser.add_argument(
        "--min-duration-on",
        type=float,
        default=DEFAULT_MIN_DURATION_ON,
        help="Remove predicted overlap segments shorter than this many seconds.",
    )
    parser.add_argument(
        "--min-duration-off",
        type=float,
        default=DEFAULT_MIN_DURATION_OFF,
        help="Fill non-overlap gaps shorter than this many seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing OSD JSON files.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target sample rate for the model.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    run_detection(
        input_path=args.input,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        sample_rate=args.sample_rate,
        onset=args.onset,
        offset=args.offset,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
        force=args.force,
    )

    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

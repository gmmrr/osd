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
DEFAULT_OVERLAP_THRESHOLD = 0.5
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU.")
        return "cpu"
    return device


def _collect_audio_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES)
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
        resampled = []
        for channel in waveform:
            resampled.append(scipy.signal.resample_poly(channel, sample_rate, sr).astype(np.float32))
        max_len = max((len(channel) for channel in resampled), default=0)
        waveform = np.stack(
            [np.pad(channel, (0, max_len - len(channel)), mode="constant") for channel in resampled],
            axis=0,
        )

    if waveform.shape[0] > 1:
        waveform = waveform[:1]

    return {
        "waveform": torch.from_numpy(np.asarray(waveform, dtype=np.float32)),
        "sample_rate": sample_rate,
    }


def _load_model(model_path: Path, device: str):
    _allow_pyannote_checkpoint_globals()
    _patch_torch_load_weights_only()

    from pyannote.audio import Model

    if not model_path.exists():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")

    model = Model.from_pretrained(str(model_path))

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


def _scores_to_overlap_segments(scores, threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    if not hasattr(scores, "data") or scores.data.ndim != 2:
        return segments

    data = np.asarray(scores.data)
    window = scores.sliding_window
    active = (data >= threshold).sum(axis=1) >= 2

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
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    force: bool = False,
) -> tuple[Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / f"{input_path.stem}_osd.json"
    if out_json.exists():
        if not force:
            return out_json, True

    audio = _load_audio_in_memory(input_path, sample_rate=sample_rate)
    result = detector(audio)

    overlaps = _scores_to_overlap_segments(result, threshold=threshold)
    overlap_duration = round(sum(item["duration"] for item in overlaps), 3)
    input_duration = round(float(audio["waveform"].shape[-1]) / sample_rate, 3)

    payload = {
        "uri": input_path.stem,
        "input": str(input_path.resolve()),
        "model_path": str(DEFAULT_MODEL_PATH.resolve()),
        "pipeline_name": DEFAULT_PIPELINE_NAME,
        "sample_rate": sample_rate,
        "overlap_threshold": threshold,
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
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    force: bool = False,
) -> tuple[int, int]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    model_path = Path(model_path)

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
            sample_rate=sample_rate,
            threshold=threshold,
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
        help="Overlap score threshold. Lower values are more permissive.",
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
        threshold=args.threshold,
        force=args.force,
    )

    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Cornell OSDC/TCN-COUNT checkpoints on WAV files.

The JSON output intentionally follows ``repo_root/osdc/osdc.py``.  Cornell's
model is different, however: it consumes 80-dimensional Kaldi MFCC features
and returns one five-class speaker-count prediction per feature frame.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import yaml
from scipy.signal import get_window
from torch import Tensor, nn
from torchaudio.compliance.kaldi import mfcc


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_LABELS = ("0", "1", "2", "3", "4+")
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_WINDOW_SIZE = 400
DEFAULT_LOOKAHEAD = 200
DEFAULT_LOOKBEHIND = 200


def _load_tcn_module() -> ModuleType:
    """Load the bundled TCN without confusing this file with package ``osdc``."""

    source = PROJECT_ROOT / "osdc" / "models" / "tcn.py"
    spec = importlib.util.spec_from_file_location("cornell_osdc_tcn", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Cornell TCN implementation from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TCN = _load_tcn_module().TCN


@dataclass(frozen=True)
class DetectionSummary:
    files: int
    skipped: int
    sample_rate: int | None


@dataclass(frozen=True)
class CornellScores:
    """Frame probabilities and their time geometry."""

    data: np.ndarray
    timestamps: np.ndarray
    duration: float


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def discover_input_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {path}")
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".wav")


def resolve_checkpoint(model_path: Path) -> Path:
    """Accept either a checkpoint file or an experiment/checkpoint directory."""

    if model_path.is_file():
        return model_path
    candidates = (model_path / "checkpoints" / "last.ckpt", model_path / "last.ckpt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Checkpoint not found: {model_path}. Expected a .ckpt file or a directory containing checkpoints/last.ckpt."
    )


def resolve_config(checkpoint: Path, config_path: Path | None) -> Path:
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return config_path
    candidates = (checkpoint.parent.parent / "confs.yml", checkpoint.parent / "confs.yml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not infer confs.yml from the checkpoint. Pass the experiment configuration with --config."
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    feature_config = config.get("feats", {})
    if feature_config.get("type") != "mfcc_kaldi":
        raise ValueError(f"This inference program requires feats.type=mfcc_kaldi; got {feature_config.get('type')!r}")
    if int(feature_config.get("n_feats", -1)) != 80:
        raise ValueError(f"Cornell TCN expects 80 input features; got {feature_config.get('n_feats')!r}")
    if int(config.get("data", {}).get("n_classes", -1)) != len(CLASS_LABELS):
        raise ValueError(f"Cornell TCN requires {len(CLASS_LABELS)} output classes")
    return config


def load_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, float]:
    waveform, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = waveform.mean(axis=1)
    if source_rate != sample_rate:
        factor = gcd(sample_rate, source_rate)
        mono = scipy.signal.resample_poly(mono, sample_rate // factor, source_rate // factor).astype(np.float32)
    duration = float(len(mono)) / sample_rate
    return mono.astype(np.float32, copy=False), duration


def _frame_indices(data_length: int, size: int, step: int) -> list[tuple[int, int]]:
    """Match Cornell's windowed feature extraction, including its 20 ms overlap."""

    count = int((data_length - size + step) / step)
    result = [(index * step, index * step + size) for index in range(max(0, count))]
    last_index = count - 1
    if last_index * step + size < data_length and data_length - (last_index + 1) * step > 0:
        result.append(((last_index + 1) * step, data_length))
    return result


def extract_features(waveform: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Extract exactly the Kaldi MFCC/filterbank representation used in training."""

    mfcc_config = dict(config["mfcc_kaldi"])
    sample_rate = int(mfcc_config.get("sample_frequency", DEFAULT_SAMPLE_RATE))
    frame_length = int(round(sample_rate * float(mfcc_config.get("frame_length", 25.0)) / 1000.0))
    if waveform.size < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.size))

    window_size = sample_rate * 30
    step = window_size - 320  # two 10 ms frames; prevents loss at chunk boundaries
    chunks: list[np.ndarray] = []
    for start, end in _frame_indices(len(waveform), window_size, step):
        chunk = torch.from_numpy(waveform[start:end].reshape(1, -1))
        feature = mfcc(chunk, **mfcc_config).transpose(0, 1)
        chunks.append(feature.numpy())
    if not chunks:
        raise ValueError("Audio is too short to extract a feature frame")
    features = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)
    if features.ndim != 2 or features.shape[0] != 80:
        raise ValueError(f"Expected features with shape [80, frames], got {features.shape}")
    return features


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    """Recreate the architecture from the AMI training script and load its Lightning state."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed checkpoint: expected a mapping in {checkpoint}")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError(f"Malformed checkpoint: missing state_dict in {checkpoint}")

    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not model_state:
        # Also support a state dict saved directly from the TCN module.
        model_state = {key: value for key, value in state.items() if key != "loss_fn.weight"}

    model = TCN(
        in_chan=80,
        n_src=5,
        out_chan=1,
        n_blocks=5,
        n_repeats=3,
        bn_chan=64,
        hid_chan=128,
    )
    model.load_state_dict(model_state, strict=True)
    return model.eval().to(device)


def overlap_add_logits(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    window_size: int,
    lookahead: int,
    lookbehind: int,
) -> np.ndarray:
    """Infer overlapping feature windows and return averaged logits [5, frames]."""

    if window_size <= 0 or window_size % 2:
        raise ValueError("window_size must be a positive even integer")
    if lookahead < 0 or lookbehind < 0:
        raise ValueError("lookahead and lookbehind must be non-negative")

    num_frames = features.shape[-1]
    stride = window_size // 2
    weights = get_window("hann", window_size, fftbins=True).astype(np.float32)
    logits_sum = np.zeros((len(CLASS_LABELS), num_frames), dtype=np.float32)
    weight_sum = np.zeros(num_frames, dtype=np.float32)

    # Starting one half-window before frame zero makes the Hann window nonzero
    # at both recording boundaries, like Cornell's left-padded overlap-add.
    for output_start in range(-stride, num_frames, stride):
        context_start = output_start - lookbehind
        context_end = output_start + window_size + lookahead
        source_start = max(0, context_start)
        source_end = min(num_frames, context_end)
        left_pad = source_start - context_start
        right_pad = context_end - source_end
        context = np.pad(features[:, source_start:source_end], ((0, 0), (left_pad, right_pad)))

        tensor = torch.from_numpy(context).unsqueeze(0).to(device)
        with torch.inference_mode():
            prediction: Tensor = model(tensor)
        central = prediction[0, :, lookbehind : lookbehind + window_size].float().cpu().numpy()
        if central.shape != (len(CLASS_LABELS), window_size):
            raise ValueError(f"Unexpected TCN output shape {tuple(prediction.shape)}")

        valid_start = max(0, output_start)
        valid_end = min(num_frames, output_start + window_size)
        if valid_end <= valid_start:
            continue
        window_start = valid_start - output_start
        window_end = window_start + valid_end - valid_start
        selected_weights = weights[window_start:window_end]
        logits_sum[:, valid_start:valid_end] += central[:, window_start:window_end] * selected_weights
        weight_sum[valid_start:valid_end] += selected_weights

    if np.any(weight_sum <= 0):
        raise RuntimeError("Overlap-add left one or more feature frames without a prediction")
    return logits_sum / weight_sum[None, :]


def infer_scores(
    model: nn.Module,
    waveform: np.ndarray,
    duration: float,
    config: Mapping[str, Any],
    device: torch.device,
    window_size: int,
    lookahead: int,
    lookbehind: int,
) -> CornellScores:
    features = extract_features(waveform, config)
    logits = overlap_add_logits(model, features, device, window_size, lookahead, lookbehind)
    probabilities = torch.from_numpy(logits.T).softmax(dim=-1).numpy()

    mfcc_config = config["mfcc_kaldi"]
    frame_shift = float(mfcc_config.get("frame_shift", 10.0)) / 1000.0
    frame_length = float(mfcc_config.get("frame_length", 25.0)) / 1000.0
    timestamps = frame_length / 2.0 + np.arange(probabilities.shape[0], dtype=np.float64) * frame_shift
    return CornellScores(data=probabilities, timestamps=timestamps, duration=duration)


def scores_to_segments(scores: CornellScores) -> list[dict[str, object]]:
    data = np.asarray(scores.data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != len(CLASS_LABELS):
        raise ValueError(f"Expected scores with shape [frames, {len(CLASS_LABELS)}], got {data.shape}")
    if data.shape[0] == 0:
        return []

    counts = data.argmax(axis=1)
    edges = np.empty(len(counts) + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = scores.duration
    edges[1:-1] = 0.5 * (scores.timestamps[:-1] + scores.timestamps[1:])

    segments: list[dict[str, object]] = []
    start_index = 0
    for index in range(1, len(counts) + 1):
        if index == len(counts) or counts[index] != counts[start_index]:
            count = int(counts[start_index])
            start = float(edges[start_index])
            end = float(edges[index])
            segments.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "count": count,
                    "label": CLASS_LABELS[count],
                    "confidence": round(float(data[start_index:index, count].mean()), 6),
                }
            )
            start_index = index
    return segments


def detect_file(
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    input_path: Path,
    output_dir: Path,
    model_path: Path,
    sample_rate: int,
    force: bool,
    window_size: int,
    lookahead: int,
    lookbehind: int,
) -> tuple[Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_osdc.json"
    if output_path.exists() and not force:
        return output_path, True

    waveform, duration = load_audio(input_path, sample_rate)
    scores = infer_scores(model, waveform, duration, config, device, window_size, lookahead, lookbehind)
    counts = scores_to_segments(scores)
    payload = {
        "uri": input_path.stem,
        "input": str(input_path.resolve()),
        "model_path": str(model_path.resolve()),
        "pipeline_name": "CornellTCN",
        "sample_rate": sample_rate,
        "classes": list(CLASS_LABELS),
        "input_duration": round(duration, 3),
        "num_count_segments": len(counts),
        "counts": counts,
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    return output_path, False


def run_detection(
    input_dir: Path | str,
    output_dir: Path | str,
    model_path: Path | str,
    config_path: Path | str | None = None,
    device: str = "auto",
    force: bool = False,
    window_size: int = DEFAULT_WINDOW_SIZE,
    lookahead: int = DEFAULT_LOOKAHEAD,
    lookbehind: int = DEFAULT_LOOKBEHIND,
) -> DetectionSummary:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    checkpoint = resolve_checkpoint(Path(model_path))
    config_file = resolve_config(checkpoint, Path(config_path) if config_path is not None else None)
    config = load_config(config_file)
    sample_rate = int(config["mfcc_kaldi"].get("sample_frequency", DEFAULT_SAMPLE_RATE))
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(f"This Cornell TCN was trained for {DEFAULT_SAMPLE_RATE} Hz audio, got {sample_rate}")

    input_files = discover_input_files(input_dir)
    if not input_files:
        return DetectionSummary(files=0, skipped=0, sample_rate=None)

    selected_device = resolve_device(device)
    model = load_model(checkpoint, selected_device)
    skipped = 0
    total = len(input_files)
    for index, input_path in enumerate(input_files, start=1):
        _, did_skip = detect_file(
            model=model,
            config=config,
            device=selected_device,
            input_path=input_path,
            output_dir=output_dir,
            model_path=checkpoint,
            sample_rate=sample_rate,
            force=force,
            window_size=window_size,
            lookahead=lookahead,
            lookbehind=lookbehind,
        )
        skipped += int(did_skip)
        print(f"Progress: {100.0 * index / total:6.2f}% ({index}/{total}) | Skipped: {skipped}/{total}", end="\r", flush=True)
    print()
    return DetectionSummary(files=total, skipped=skipped, sample_rate=sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cornell five-class TCN-COUNT inference and output JSON.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Checkpoint file or experiment directory")
    parser.add_argument("--config", type=Path, help="confs.yml; inferred from the experiment directory by default")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="Inference window in feature frames")
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD, help="Right context in feature frames")
    parser.add_argument("--lookbehind", type=int, default=DEFAULT_LOOKBEHIND, help="Left context in feature frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_detection(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        config_path=args.config,
        device=args.device,
        force=args.force,
        window_size=args.window_size,
        lookahead=args.lookahead,
        lookbehind=args.lookbehind,
    )
    if summary.files == 0:
        print(f"⚠️  No WAV files found in: {args.input_dir}")
        return
    print(f"🚀 Cornell OSDC in '{args.input_dir}'")
    print(f"   • Files: {summary.files}")
    print(f"   • Sample rate: {summary.sample_rate}")
    print(f"   • Model: {args.model_path}")
    print("✅ Cornell OSDC completed.")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()

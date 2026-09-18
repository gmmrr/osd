#!/usr/bin/env python3
"""Native model, data, checkpoint, and inference contracts for OSDC."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from math import gcd
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from pyannote.audio.models.blocks.sincnet import SincNet
from torch import Tensor, nn
from torch.utils.data import Dataset

NUM_CLASSES = 5
MAX_ACTIVE_SPEAKERS = NUM_CLASSES - 1
DEFAULT_SAMPLE_RATE = 16000
CHECKPOINT_SCHEMA_VERSION = 1
CLASS_LABELS = tuple(str(count) for count in range(MAX_ACTIVE_SPEAKERS)) + (f"{MAX_ACTIVE_SPEAKERS}+",)

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "sample_rate": DEFAULT_SAMPLE_RATE,
    "num_channels": 1,
    "sincnet": {"stride": 10},
    "lstm": {"hidden_size": 128, "num_layers": 4, "bidirectional": True, "dropout": 0.5, "monolithic": True},
    "linear": {"hidden_size": 128, "num_layers": 2},
    "num_classes": NUM_CLASSES,
}


@dataclass(frozen=True)
class RTTMSegment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Chunk:
    uri: str
    audio_path: Path
    start: float
    duration: float
    valid_duration: float


@dataclass(frozen=True)
class CountScores:
    data: np.ndarray
    timestamps: np.ndarray
    duration: float


def _parse_lst(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as stream:
        return [line.strip().split()[0] for line in stream if line.strip() and not line.lstrip().startswith("#")]


def _parse_rttm(path: Path) -> dict[str, list[RTTMSegment]]:
    result: dict[str, list[RTTMSegment]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8 or fields[0].upper() != "SPEAKER":
                raise ValueError(f"Malformed RTTM line {line_number} in {path}: {line.rstrip()}")
            start, duration = float(fields[3]), float(fields[4])
            if duration > 0:
                result[fields[1]].append(RTTMSegment(start, start + duration, fields[7]))
    return dict(result)


def _parse_uem(path: Path) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed UEM line {line_number} in {path}: {line.rstrip()}")
            start, end = float(fields[2]), float(fields[3])
            if end > start:
                result[fields[0]].append((start, end))
    return dict(result)


def speaker_count_target(
    segments: Sequence[RTTMSegment], chunk_start: float, frame_center: float, frame_step: float, num_frames: int
) -> Tensor:
    """Rasterize RTTM activity and clip counts to the highest class.

    The rounding convention matches pyannote SpeakerDiarization.prepare_chunk.
    Identity is used only to avoid counting overlapping segments from one
    speaker twice; it never becomes a target class.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")

    active_by_speaker: dict[str, np.ndarray] = {}
    last_center = chunk_start + frame_center + (num_frames - 1) * frame_step
    for segment in segments:
        if segment.start > last_center + frame_center or segment.end <= chunk_start:
            continue
        start = max(segment.start, chunk_start) - chunk_start - frame_center
        end = min(segment.end, last_center + frame_center) - chunk_start - frame_center
        start_idx = max(0, int(np.round(start / frame_step)))
        end_idx = min(num_frames - 1, int(np.round(end / frame_step)))
        if end_idx < 0 or start_idx >= num_frames or end_idx < start_idx:
            continue
        active = active_by_speaker.setdefault(segment.speaker, np.zeros(num_frames, dtype=np.bool_))
        active[start_idx : end_idx + 1] = True

    if active_by_speaker:
        counts = np.stack(list(active_by_speaker.values()), axis=0).sum(axis=0)
    else:
        counts = np.zeros(num_frames, dtype=np.int64)
    return torch.from_numpy(np.minimum(counts, MAX_ACTIVE_SPEAKERS).astype(np.int64))


def _audio_duration(path: Path) -> float:
    return float(sf.info(str(path)).duration)


def _chunk_regions(start: float, end: float, stride: float) -> Iterable[float]:
    position = start
    while position < end - 1e-9:
        yield position
        position += stride


class OSDCChunkDataset(Dataset[dict[str, Tensor | str | float]]):
    """RTTM/UEM-backed chunks with frame-level speaker-count targets."""

    def __init__(
        self,
        dataset_root: Path | str,
        split: str,
        chunk_duration: float,
        num_frames: int,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_center: float | None = None,
        frame_step: float | None = None,
        frame_duration: float | None = None,
        stride: float | None = None,
    ) -> None:
        super().__init__()
        if chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive")
        self.root = Path(dataset_root)
        self.split = split
        self.chunk_duration = float(chunk_duration)
        self.num_samples = round(self.chunk_duration * sample_rate)
        self.num_frames = int(num_frames)
        self.sample_rate = sample_rate
        self.frame_step = float(frame_step or self.chunk_duration / self.num_frames)
        self.frame_center = float(frame_center if frame_center is not None else 0.5 * self.frame_step)
        self.frame_duration = float(frame_duration or 2.0 * self.frame_center)
        stride = float(stride or chunk_duration)

        uris = _parse_lst(self.root / "lists" / f"{split}.lst")
        self.segments = _parse_rttm(self.root / "rttm" / f"{split}.rttm")
        uem_path = self.root / "uem" / f"{split}.uem"
        regions = _parse_uem(uem_path) if uem_path.exists() else {}
        self.chunks: list[Chunk] = []
        for uri in uris:
            audio_path = self.root / "audio" / f"{uri}.wav"
            for region_start, region_end in regions.get(uri, [(0.0, _audio_duration(audio_path))]):
                for chunk_start in _chunk_regions(region_start, region_end, stride):
                    chunk = Chunk(uri, audio_path, chunk_start, self.chunk_duration, min(self.chunk_duration, region_end - chunk_start))
                    if not self.valid_for_chunk(chunk).any():
                        continue
                    self.chunks.append(chunk)

    def __len__(self) -> int:
        return len(self.chunks)

    def target_for_chunk(self, chunk: Chunk) -> Tensor:
        return speaker_count_target(
            self.segments.get(chunk.uri, ()), chunk.start, self.frame_center, self.frame_step, self.num_frames
        )

    def valid_for_chunk(self, chunk: Chunk) -> Tensor:
        centres = self.frame_center + torch.arange(self.num_frames, dtype=torch.float64) * self.frame_step
        return centres + 0.5 * self.frame_duration <= chunk.valid_duration + 1e-9

    def class_counts(self) -> Tensor:
        counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        for chunk in self.chunks:
            target = self.target_for_chunk(chunk)
            valid = self.valid_for_chunk(chunk)
            if valid.any():
                counts += torch.bincount(target[valid], minlength=NUM_CLASSES)
        return counts

    def __getitem__(self, index: int) -> dict[str, Tensor | str | float]:
        chunk = self.chunks[index]
        info = sf.info(str(chunk.audio_path))
        start_frame = max(0, round(chunk.start * info.samplerate))
        waveform, source_rate = sf.read(
            str(chunk.audio_path), start=start_frame, frames=round(chunk.duration * info.samplerate), dtype="float32", always_2d=True
        )
        waveform = waveform.mean(axis=1)
        if source_rate != self.sample_rate:
            factor = gcd(source_rate, self.sample_rate)
            waveform = scipy.signal.resample_poly(waveform, self.sample_rate // factor, source_rate // factor).astype(np.float32)
        samples = torch.from_numpy(np.asarray(waveform, dtype=np.float32))[: self.num_samples]
        samples = F.pad(samples, (0, max(0, self.num_samples - samples.numel())))
        return {
            "waveform": samples.unsqueeze(0), "target": self.target_for_chunk(chunk), "valid": self.valid_for_chunk(chunk),
            "uri": chunk.uri, "start": chunk.start,
        }


def classification_metrics(prediction: Tensor, target: Tensor, valid: Tensor | None = None) -> dict[str, Tensor]:
    """Return confusion matrix and class-balanced frame metrics."""
    if prediction.ndim == target.ndim + 1:
        prediction = prediction.argmax(dim=-1)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch: {prediction.shape} != {target.shape}")
    valid = torch.ones_like(target, dtype=torch.bool) if valid is None else valid.bool()
    prediction, target = prediction[valid].long(), target[valid].long()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=target.device)
    if target.numel():
        confusion = torch.bincount(target * NUM_CLASSES + prediction, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = confusion.diag().float()
    support, predicted = confusion.sum(dim=1).float(), confusion.sum(dim=0).float()
    recall, precision = tp / support.clamp_min(1), tp / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = support > 0
    return {
        "confusion_matrix": confusion,
        "accuracy": tp.sum() / confusion.sum().clamp_min(1),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "macro_f1": f1[present].mean() if present.any() else f1.new_tensor(0.0),
    }


class Model(LightningModule):
    """Five-class frame model independent of its initialization source."""

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__()
        config = _normalized_model_config(model_config)
        self.save_hyperparameters({
            "model_config": config, "learning_rate": float(learning_rate), "weight_decay": float(weight_decay),
        })
        self.model_config = config

        sincnet_config = dict(config["sincnet"])
        sincnet_config.pop("sample_rate", None)
        self.sincnet = SincNet(sample_rate=config["sample_rate"], **sincnet_config)
        lstm_config = dict(config["lstm"])
        monolithic = bool(lstm_config.pop("monolithic"))
        lstm_config["batch_first"] = True
        if monolithic:
            self.lstm: nn.LSTM | nn.ModuleList = nn.LSTM(60, **lstm_config)
        else:
            num_layers = int(lstm_config.pop("num_layers"))
            dropout = float(lstm_config.pop("dropout"))
            self.dropout = nn.Dropout(dropout)
            self.lstm = nn.ModuleList()
            for index in range(num_layers):
                input_size = 60 if index == 0 else lstm_config["hidden_size"] * (2 if lstm_config["bidirectional"] else 1)
                self.lstm.append(nn.LSTM(input_size, num_layers=1, dropout=0.0, **lstm_config))

        linear_config = config["linear"]
        lstm_features = config["lstm"]["hidden_size"] * (2 if config["lstm"]["bidirectional"] else 1)
        sizes = [lstm_features] + [linear_config["hidden_size"]] * int(linear_config["num_layers"])
        self.linear = nn.ModuleList(nn.Linear(a, b) for a, b in pairwise(sizes))
        self.count_head = nn.Linear(sizes[-1], NUM_CLASSES)
        weights = torch.as_tensor(class_weights if class_weights is not None else torch.ones(NUM_CLASSES), dtype=torch.float32)
        if weights.numel() != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} class weights, got {weights.numel()}")
        self.register_buffer("class_weights", weights, persistent=False)

    @property
    def sample_rate(self) -> int:
        return int(self.model_config["sample_rate"])

    def num_frames(self, num_samples: int) -> int:
        return self.sincnet.num_frames(num_samples)

    def frame_geometry(self) -> tuple[float, float, float]:
        size = self.sincnet.receptive_field_size(num_frames=1)
        step = self.sincnet.receptive_field_size(num_frames=2) - size
        center = self.sincnet.receptive_field_center(frame=0)
        # SlidingWindow.middle uses start + duration / 2, i.e. half a sample
        # after the integer receptive-field centre for an odd-sized window.
        return (center + 0.5) / self.sample_rate, step / self.sample_rate, size / self.sample_rate

    def extract_features(self, waveform: Tensor) -> Tensor:
        outputs = self.sincnet(waveform).transpose(1, 2)
        if isinstance(self.lstm, nn.LSTM):
            outputs, _ = self.lstm(outputs)
        else:
            for index, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if index + 1 < len(self.lstm):
                    outputs = self.dropout(outputs)
        for linear in self.linear:
            outputs = F.leaky_relu(linear(outputs))
        return outputs

    def forward(self, waveform: Tensor) -> Tensor:
        """Return raw frame logits with shape [batch, frames, classes]."""
        return self.count_head(self.extract_features(waveform))

    def predict(self, waveform: Tensor, *, probabilities: bool = False) -> Tensor:
        logits = self(waveform)
        return logits.softmax(dim=-1) if probabilities else logits.argmax(dim=-1)

    def compute_loss(self, logits: Tensor, target: Tensor, valid: Tensor | None = None) -> Tensor:
        if logits.shape[:-1] != target.shape or logits.shape[-1] != NUM_CLASSES:
            raise ValueError(f"Expected logits [batch, frames, {NUM_CLASSES}] matching target; got {logits.shape} and {target.shape}")
        valid = torch.ones_like(target, dtype=torch.bool) if valid is None else valid.bool()
        if not valid.any():
            raise ValueError("Batch contains no valid frames")
        return F.cross_entropy(logits[valid], target[valid], weight=self.class_weights)

    def evaluate(self, logits: Tensor, target: Tensor, valid: Tensor | None = None) -> dict[str, Tensor]:
        return classification_metrics(logits, target, valid)

    def _shared_step(self, batch: dict[str, Tensor], prefix: str) -> Tensor:
        logits = self(batch["waveform"])
        target, valid = batch["target"], batch["valid"]
        loss = self.compute_loss(logits, target, valid)
        metrics = self.evaluate(logits.detach(), target, valid)
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=prefix == "train", on_epoch=True, batch_size=target.shape[0])
        self.log(f"{prefix}_accuracy", metrics["accuracy"], on_step=False, on_epoch=True, batch_size=target.shape[0])
        self.log(f"{prefix}_macro_f1", metrics["macro_f1"], prog_bar=prefix == "val", on_step=False, on_epoch=True, batch_size=target.shape[0])
        for index, value in enumerate(metrics["per_class_recall"]):
            self.log(f"{prefix}_recall_{CLASS_LABELS[index].replace('+', 'plus')}", value, on_step=False, on_epoch=True, batch_size=target.shape[0])
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["osdc"] = self.checkpoint_metadata()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        _validate_checkpoint(checkpoint)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION, "task": "frame-level-speaker-count",
            "classes": list(CLASS_LABELS), "model_config": self.model_config,
        }

    @classmethod
    def from_checkpoint(cls, path: Path | str, map_location: str | torch.device = "cpu", **overrides: Any) -> "Model":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        _validate_checkpoint(checkpoint)
        hparams = dict(checkpoint.get("hyper_parameters", {}))
        hparams.update(overrides)
        model = cls(**hparams)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model


def _normalized_model_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_MODEL_CONFIG if config is None else config
    result = {
        "sample_rate": int(source.get("sample_rate", DEFAULT_SAMPLE_RATE)), "num_channels": int(source.get("num_channels", 1)),
        "sincnet": {**DEFAULT_MODEL_CONFIG["sincnet"], **dict(source.get("sincnet", {}))},
        "lstm": {**DEFAULT_MODEL_CONFIG["lstm"], **dict(source.get("lstm", {}))},
        "linear": {**DEFAULT_MODEL_CONFIG["linear"], **dict(source.get("linear", {}))},
        "num_classes": int(source.get("num_classes", NUM_CLASSES)),
    }
    if result["sample_rate"] != DEFAULT_SAMPLE_RATE or result["num_channels"] != 1:
        raise ValueError("The current OSDC frontend requires 16 kHz mono audio")
    if result["num_classes"] != NUM_CLASSES:
        raise ValueError(f"OSDC requires exactly {NUM_CLASSES} output classes")
    return result


def _validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    metadata = checkpoint.get("osdc")
    if not isinstance(metadata, Mapping):
        raise ValueError("Not a native OSDC checkpoint: missing 'osdc' metadata")
    version = metadata.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported OSDC checkpoint schema {version!r}; expected {CHECKPOINT_SCHEMA_VERSION}")
    if tuple(metadata.get("classes", ())) != CLASS_LABELS:
        raise ValueError(f"Checkpoint classes must be {CLASS_LABELS}")


def save_native_checkpoint(model: Model, path: Path | str) -> Path:
    """Save a Lightning-compatible native OSDC initialization checkpoint."""
    import lightning.pytorch as pl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": pl.__version__, "osdc": model.checkpoint_metadata(),
    }, path)
    return path


class Inference:
    """Sliding-window inference shared by both initialization routes."""

    def __init__(self, model: Model, device: torch.device, chunk_duration: float = 10.0) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.chunk_samples = round(chunk_duration * model.sample_rate)
        self.frame_center, self.frame_step, self.frame_duration = model.frame_geometry()
        self.stride_frames = max(1, round(0.5 * chunk_duration / self.frame_step))
        self.stride_samples = round(self.stride_frames * self.frame_step * model.sample_rate)

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @torch.inference_mode()
    def __call__(self, audio: Mapping[str, object]) -> CountScores:
        waveform = audio["waveform"]
        samples = waveform.shape[-1]
        duration = samples / self.model.sample_rate
        num_frames = max(0, self.model.num_frames(samples))
        probabilities_sum = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
        probabilities_count = np.zeros(num_frames, dtype=np.float32)
        for chunk_index, start in enumerate(range(0, max(samples, 1), self.stride_samples)):
            chunk = waveform[:, start : start + self.chunk_samples]
            valid_samples = min(self.chunk_samples, max(0, samples - start))
            chunk = F.pad(chunk, (0, self.chunk_samples - chunk.shape[-1])).unsqueeze(0).to(self.device)
            probabilities = self.model.predict(chunk, probabilities=True).squeeze(0).cpu().numpy()
            valid_duration = valid_samples / self.model.sample_rate
            centres = self.frame_center + np.arange(len(probabilities)) * self.frame_step
            valid_frames = centres + 0.5 * self.frame_duration <= valid_duration + 1e-9
            start_frame = chunk_index * self.stride_frames
            end_frame = min(num_frames, start_frame + int(valid_frames.sum()))
            if end_frame > start_frame:
                probabilities_sum[start_frame:end_frame] += probabilities[: end_frame - start_frame]
                probabilities_count[start_frame:end_frame] += 1.0
        data = probabilities_sum / np.maximum(probabilities_count[:, None], 1.0)
        timestamps = self.frame_center + np.arange(num_frames, dtype=np.float64) * self.frame_step
        return CountScores(data=data, timestamps=timestamps, duration=duration)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune the local WeSpeaker embedding model on NvvMix speaker labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.protocol.speaker_diarization import SpeakerDiarizationProtocol
from pyannote.database.util import load_lst, load_rttm, load_uem


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "diarization/sd3/models/wespeaker-voxceleb-resnet34-LM"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "diarization/sd3/models/wespeaker-train-nvv-v1"
DEFAULT_CHUNK_DURATION = 2.0
DEFAULT_MIN_CHUNK_DURATION = 0.5
SPLITS = ("train", "dev", "test")


def resolve_path(path: Path) -> Path:
    """Accept an absolute path or a path relative to the repository root."""
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def resolve_checkpoint(path: Path) -> Path:
    """Resolve a pyannote model directory or checkpoint file."""
    path = resolve_path(path)
    checkpoint = path / "pytorch_model.bin" if path.is_dir() else path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"WeSpeaker checkpoint not found: {checkpoint}")
    return checkpoint


def exclusive_annotation(annotation: Annotation, annotated: Timeline, min_duration: float) -> Annotation:
    """Retain only speaker turns without another annotated speaker on top."""
    clean = Annotation(uri=annotation.uri)
    labels = annotation.labels()
    for speaker in labels:
        own = annotation.label_timeline(speaker).support()
        other = Timeline()
        for label in labels:
            if label != speaker:
                other.update(annotation.label_timeline(label))
        for segment in own.extrude(other.support()).crop(annotated).support():
            if segment.duration > min_duration:
                # RTTM labels have file scope; never merge equal names across mixes.
                clean[segment] = f"{annotation.uri}:{speaker}"
    return clean


def build_protocol(dataset_root: Path, min_duration: float = DEFAULT_MIN_CHUNK_DURATION) -> SpeakerDiarizationProtocol:
    """Read the repository's NvvMix audio/lists/RTTM/UEM split layout."""
    dataset_root = resolve_path(dataset_root)
    for split in SPLITS:
        for directory, suffix in (("lists", "lst"), ("rttm", "rttm"), ("uem", "uem")):
            path = dataset_root / directory / f"{split}.{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"Missing NvvMix {split} file: {path}")

    class NvvMixProtocol(SpeakerDiarizationProtocol):
        def __init__(self) -> None:
            super().__init__()
            self.name = "NvvMix.SpeakerDiarization.WeSpeaker"
            self._splits = {
                split: {
                    "uris": load_lst(dataset_root / "lists" / f"{split}.lst"),
                    "annotations": load_rttm(dataset_root / "rttm" / f"{split}.rttm"),
                    "annotated": load_uem(dataset_root / "uem" / f"{split}.uem"),
                }
                for split in SPLITS
            }

        def _iter(self, split: str):
            data = self._splits[split]
            for uri in data["uris"]:
                if uri not in data["annotations"] or uri not in data["annotated"]:
                    raise ValueError(f"Missing {split} RTTM or UEM for {uri}")
                audio = dataset_root / "audio" / f"{uri}.wav"
                if not audio.is_file():
                    raise FileNotFoundError(f"Missing {split} audio: {audio}")
                annotated = data["annotated"][uri]
                yield {
                    "uri": uri,
                    "database": "NvvMix",
                    "subset": "development" if split == "dev" else split,
                    "scope": "file",
                    "audio": str(audio),
                    "annotation": exclusive_annotation(data["annotations"][uri], annotated, min_duration),
                    "annotated": annotated,
                }

        def train_iter(self):
            return self._iter("train")

        def development_iter(self):
            return self._iter("dev")

        def test_iter(self):
            return self._iter("test")

    return NvvMixProtocol()


def validation_pairs(protocol: SpeakerDiarizationProtocol) -> list[tuple[dict, Segment, dict, Segment, int]]:
    """Create same/different speaker pairs from clean dev turns in each recording."""
    pairs = []
    for file in protocol.development():
        turns = {
            label: list(file["annotation"].label_timeline(label).support())
            for label in file["annotation"].labels()
        }
        labels = sorted(turns)
        for label in labels:
            if len(turns[label]) < 2:
                continue
            pairs.append((file, turns[label][0], file, turns[label][1], 1))
            other = next((item for item in labels if item != label and turns[item]), None)
            if other is not None:
                pairs.append((file, turns[label][0], file, turns[other][0], 0))
    if not any(pair[-1] == 1 for pair in pairs) or not any(pair[-1] == 0 for pair in pairs):
        raise ValueError("NvvMix dev needs both same-speaker and different-speaker clean turn pairs.")
    return pairs


def resolve_accelerator(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA unavailable; using CPU.")
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS unavailable; using CPU.")
        return "cpu"
    return device


def export_embedding(model: Any, output_dir: Path, epoch: int, step: int, checkpoint_path: Path | None = None) -> Path:
    """Write a pyannote checkpoint without the training-only ArcFace classifier."""
    import lightning.pytorch as pl

    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path else None
    state = source["state_dict"] if source else model.state_dict()
    export = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in state.items()
            if not key.startswith("loss_func.")
        },
        "pytorch-lightning_version": pl.__version__,
        "hyper_parameters": dict(model.hparams),
        "epoch": source["epoch"] if source else epoch,
        "global_step": source["global_step"] if source else step,
    }
    if source:
        export["pyannote.audio"] = source["pyannote.audio"]
    else:
        model.on_save_checkpoint(export)
    path = output_dir / "pytorch_model.bin"
    torch.save(export, path)
    return path


def run_training(args: argparse.Namespace) -> Path:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from pyannote.audio import Model
    from pyannote.audio.tasks import SupervisedRepresentationLearningWithArcFace
    from torch.nn import functional as F

    model_path = resolve_checkpoint(args.model)
    output_dir = resolve_path(args.output_dir)
    if output_dir == DEFAULT_MODEL or DEFAULT_MODEL in output_dir.parents:
        raise ValueError("Refusing to overwrite the original WeSpeaker model.")
    if output_dir.parent != DEFAULT_MODEL.parent or not output_dir.name.startswith("wespeaker-train-"):
        raise ValueError(f"Output must be a wespeaker-train-* directory under {DEFAULT_MODEL.parent}")
    if (output_dir / "pytorch_model.bin").exists() and args.resume is None:
        raise FileExistsError(f"Trained WeSpeaker already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    protocol = build_protocol(args.ground_truth, args.min_chunk_duration)
    pairs = validation_pairs(protocol)

    class NvvMixArcFace(SupervisedRepresentationLearningWithArcFace):
        def training_step(self, batch, batch_idx: int):
            result = super().training_step(batch, batch_idx)
            if result is not None:
                self.model.log("train_loss", result["loss"], on_step=False, on_epoch=True, prog_bar=True)
            return result

        def val__len__(self):
            return len(pairs)

        def _crop(self, file: dict, segment: Segment) -> torch.Tensor:
            duration = min(segment.duration, self.duration)
            crop = Segment(segment.start, segment.start + duration)
            waveform, _ = self.model.audio.crop(file, crop)
            size = int(self.duration * self.model.audio.sample_rate)
            return F.pad(waveform[:, :size], (0, max(0, size - waveform.shape[-1])))

        def val__getitem__(self, idx: int):
            file1, turn1, file2, turn2, same = pairs[idx]
            return {"X1": self._crop(file1, turn1), "X2": self._crop(file2, turn2), "y": same}

        def validation_step(self, batch, batch_idx: int):
            with torch.no_grad():
                similarity = F.cosine_similarity(self.model(batch["X1"]), self.model(batch["X2"]))
            self.model.validation_metric(similarity, batch["y"].int())
            self.model.log_dict(self.model.validation_metric, on_step=False, on_epoch=True, prog_bar=True)

    model = Model.from_pretrained(model_path, map_location="cpu")
    if model is None or model.__class__.__name__ != "WeSpeakerResNet34":
        raise ValueError(f"Expected a WeSpeakerResNet34 checkpoint: {model_path}")
    task = NvvMixArcFace(
        protocol,
        duration=args.chunk_duration,
        min_duration=args.min_chunk_duration,
        num_classes_per_batch=args.batch_size // 2,
        num_chunks_per_class=2,
        num_workers=args.num_workers,
    )
    model.task = task
    model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    monitor, mode = task.val_monitor
    class EpochSummary(Callback):
        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if trainer.sanity_checking:
                return
            metrics = trainer.callback_metrics
            train_loss = metrics.get("train_loss")
            validation = metrics.get(monitor)
            train_text = f"{float(train_loss):.4f}" if train_loss is not None else "n/a"
            val_text = f"{float(validation):.4f}" if validation is not None else "n/a"
            print(f"   • Epoch {trainer.current_epoch + 1}: train loss={train_text}, {monitor}={val_text}")

    callbacks = [ModelCheckpoint(dirpath=output_dir / "checkpoints", monitor=monitor, mode=mode, save_top_k=args.save_top_k, save_last=True), EpochSummary()]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=args.early_stopping_patience))

    print(f"🚀 WeSpeaker training in '{resolve_path(args.ground_truth)}'")
    print(f"   • Initialization: {model_path}")
    print(f"   • Train split: {len(load_lst(resolve_path(args.ground_truth) / 'lists/train.lst'))} recording(s)")
    print(f"   • Validation split: {len(load_lst(resolve_path(args.ground_truth) / 'lists/dev.lst'))} recording(s), {len(pairs)} pairs")
    print(f"   • Device: {resolve_accelerator(args.device)}")
    print(f"   • Epochs: {args.max_epochs}; batch size: {args.batch_size}; learning rate: {args.learning_rate}")
    print(f"   • Output: {output_dir}")

    trainer = pl.Trainer(
        accelerator=resolve_accelerator(args.device),
        devices=1,
        max_epochs=args.max_epochs,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        log_every_n_steps=1,
    )
    resume = resolve_path(args.resume) if args.resume else None
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    trainer.fit(model, ckpt_path=str(resume) if resume else None)
    if not callbacks[0].last_model_path:
        raise RuntimeError("Training completed without a checkpoint.")
    best = callbacks[0].best_model_path or callbacks[0].last_model_path
    export = export_embedding(model, output_dir, trainer.current_epoch, trainer.global_step, Path(best))
    print(f"✅ WeSpeaker training completed.\nCheckpoint: {callbacks[0].last_model_path}\nEmbedding model: {export}")
    return export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune WeSpeaker ResNet34 speaker embeddings on NvvMix.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="NvvMix root with audio/, lists/, rttm/, and uem/.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Initial WeSpeaker directory or pyannote checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="New wespeaker-train-... model directory.")
    parser.add_argument("--max-epochs", "--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8, help="Even number; two chunks per speaker per batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--chunk-duration", type=float, default=DEFAULT_CHUNK_DURATION)
    parser.add_argument("--min-chunk-duration", type=float, default=DEFAULT_MIN_CHUNK_DURATION)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--save-top-k", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--resume", type=Path, help="Resume from a Lightning checkpoint in checkpoints/.")
    args = parser.parse_args()
    if args.batch_size < 4 or args.batch_size % 2:
        parser.error("--batch-size must be even and at least 4.")
    if args.max_epochs < 1 or args.learning_rate <= 0 or args.min_chunk_duration <= 0 or args.chunk_duration < args.min_chunk_duration:
        parser.error("Epochs, learning rate, and durations must be positive, with min chunk <= chunk.")
    if args.num_workers < 0 or args.save_top_k < 1 or args.early_stopping_patience < 0:
        parser.error("Invalid workers, save-top-k, or early-stopping-patience value.")
    return args


if __name__ == "__main__":
    run_training(parse_args())

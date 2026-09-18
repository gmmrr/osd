#!/usr/bin/env python3
"""Train OSDC checkpoints initialized from segmentation-3.0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from .model import NUM_CLASSES, Model, OSDCChunkDataset

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = REPO_ROOT / "osdc" / "models" / "segmentation-3.0-based-v0"
DEFAULT_CHUNK_DURATION = 10.0 # the value when pretrained, highly recommended not to change
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_EPOCHS = 1
DEFAULT_SEED = 42 # never change it for reproduction
DEFAULT_SAVE_TOP_K = 5
DEFAULT_EARLY_STOPPING_PATIENCE = 0
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4


def run_training(
    *,
    output_dir: Path,
    dataset_root: Path,
    model_dir: Path,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    save_top_k: int,
    early_stopping_patience: int,
    device: str,
) -> Path:
    if max_epochs <= 0:
        raise ValueError("max-epochs must be positive")
    seed_everything(DEFAULT_SEED, workers=True, verbose=False)
    model_path = model_dir if model_dir.is_file() else model_dir / "checkpoints" / "last.ckpt"
    model = Model.from_checkpoint(
        model_path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    num_samples = round(DEFAULT_CHUNK_DURATION * model.sample_rate)
    num_frames = model.num_frames(num_samples)
    frame_center, frame_step, frame_duration = model.frame_geometry()
    dataset_kwargs = {
        "sample_rate": model.sample_rate, "frame_center": frame_center, "frame_step": frame_step,
        "frame_duration": frame_duration,
    }
    train_set = OSDCChunkDataset(dataset_root, "train", DEFAULT_CHUNK_DURATION, num_frames, **dataset_kwargs)
    dev_set = OSDCChunkDataset(dataset_root, "dev", DEFAULT_CHUNK_DURATION, num_frames, **dataset_kwargs)
    if not train_set:
        raise ValueError("Training split contains no chunk with a complete valid model frame")
    if not dev_set:
        raise ValueError("Development split contains no chunk with a complete valid model frame")
    counts = train_set.class_counts().float()
    class_weights = torch.zeros_like(counts)
    present = counts > 0
    class_weights[present] = counts[present].sum() / counts[present]
    if present.any():
        class_weights[present] /= class_weights[present].mean()
    model.class_weights.copy_(class_weights)
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints", filename="epoch={epoch}-val_loss={val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=save_top_k, save_last=True, auto_insert_metric_name=False,
    )
    callbacks: list[Any] = [checkpoint]
    if early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=early_stopping_patience))

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU.")
        selected_accelerator = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS requested but not available, falling back to CPU.")
        selected_accelerator = "cpu"
    elif device == "auto":
        selected_accelerator = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        selected_accelerator = device
    loader_kwargs = {
        "batch_size": batch_size, "pin_memory": selected_accelerator == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_set, shuffle=False, **loader_kwargs)

    trainer = Trainer(
        accelerator=selected_accelerator, devices=1, max_epochs=max_epochs,
        default_root_dir=str(output_dir), callbacks=callbacks, log_every_n_steps=1, deterministic=True,
        gradient_clip_val=5.0,
    )
    trainer.fit(model, train_loader, dev_loader)
    if not checkpoint.last_model_path:
        raise RuntimeError("Training finished but no checkpoint was saved.")
    return Path(checkpoint.last_model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Continue training a native {NUM_CLASSES}-class OSDC checkpoint.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--save-top-k", type=int, default=DEFAULT_SAVE_TOP_K)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"🚀 Train in '{args.ground_truth}'")
    print(f"   • Model: {args.model_dir}")
    print(f"   • Classes: {NUM_CLASSES}")
    print(f"   • Epochs this run: {args.max_epochs}")
    for split in ("train", "dev", "test"):
        lines = (args.ground_truth / "lists" / f"{split}.lst").read_text(encoding="utf-8").splitlines()
        records = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        print(f"   • {split}: {len(records)} record(s)")
    checkpoint_path = run_training(
        output_dir=args.output_dir, dataset_root=args.ground_truth, model_dir=args.model_dir,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, save_top_k=args.save_top_k,
        early_stopping_patience=args.early_stopping_patience, device=args.device,
    )
    print("✅ Training completed.")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

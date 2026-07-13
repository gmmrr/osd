#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pyannote.database.protocol.speaker_diarization import SpeakerDiarizationProtocol
from pyannote.database.util import load_lst, load_rttm, load_uem


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = REPO_ROOT / "osd" / "models" / "pyannote-segmentation-3.0"
DEFAULT_CHUNK_DURATION = 10.0 # the value when pretrained, highly recommended not to change
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_EPOCHS = 1
DEFAULT_SEED = 42 # never change it for reproduction
DEFAULT_SAVE_TOP_K = 5
DEFAULT_EARLY_STOPPING_PATIENCE = 0
DEFAULT_LEARNING_RATE = 1e-4

SPLITS = ("train", "dev", "test")
SPLIT_FILES = {
    "train": ("train.lst", "train.rttm", "train.uem"),
    "dev": ("dev.lst", "dev.rttm", "dev.uem"),
    "test": ("test.lst", "test.rttm", "test.uem"),
}


def build_protocol(dataset_root: Path) -> SpeakerDiarizationProtocol:
    class LocalDiarizationProtocol(SpeakerDiarizationProtocol):
        def __init__(self) -> None:
            super().__init__()
            self.name = "local-diarization"
            self._splits = {
                split: {
                    "uris": load_lst(dataset_root / "lists" / lst),
                    "annotations": load_rttm(dataset_root / "rttm" / rttm),
                    "annotated": load_uem(dataset_root / "uem" / uem),
                }
                for split, (lst, rttm, uem) in SPLIT_FILES.items()
            }

        def train_iter(self):
            return self._iter("train")

        def development_iter(self):
            return self._iter("dev")

        def test_iter(self):
            return self._iter("test")

        def _iter(self, split: str):
            items = self._splits[split]
            for uri in items["uris"]:
                yield {
                    "uri": uri,
                    "database": "local",
                    "scope": "file",
                    "audio": str(dataset_root / "audio" / f"{uri}.wav"),
                    "annotation": items["annotations"][uri],
                    "annotated": items["annotated"][uri],
                }

    return LocalDiarizationProtocol()


def run_training(
    output_dir: Path,
    protocol: SpeakerDiarizationProtocol,
    model_dir: Path,
    batch_size: int,
    max_epochs: int,
    device: str,
    save_top_k: int,
    early_stopping_patience: int,
    learning_rate: float,
) -> Path:
    from pyannote.audio import Model
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    import lightning.pytorch as pl
    from pyannote.audio.tasks import SpeakerDiarization
    import torch.optim

    pl.seed_everything(DEFAULT_SEED, workers=True)
    model = Model.from_pretrained(str(model_dir))
    model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), lr=learning_rate)
    task = SpeakerDiarization(
        protocol,
        duration=DEFAULT_CHUNK_DURATION,
        batch_size=batch_size,
        max_speakers_per_chunk=3,
        max_speakers_per_frame=2,
    )
    model.task = task

    if device == "cuda":
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        if accelerator == "cpu":
            print("⚠️  CUDA requested but not available, falling back to CPU.")
    elif device == "cpu":
        accelerator = "cpu"
    else:
        accelerator = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    monitor, mode = task.val_monitor
    checkpoint_kwargs = {"dirpath": str(output_dir / "checkpoints"), "save_last": True, "save_top_k": save_top_k}
    if save_top_k > 0 or early_stopping_patience > 0:
        checkpoint_kwargs.update({"monitor": monitor, "mode": mode})

    callbacks = [ModelCheckpoint(**checkpoint_kwargs)]
    if early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=early_stopping_patience))

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    last_checkpoint = callbacks[0].last_model_path
    if not last_checkpoint:
        raise RuntimeError("Training finished but no checkpoint was saved.")
    return Path(last_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pyannote segmentation-3.0 from an existing dataset root.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--save-top-k", type=int, default=DEFAULT_SAVE_TOP_K)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Local model directory not found: {args.model_dir}")

    protocol = build_protocol(args.ground_truth)

    print(f"🚀 Train in '{args.ground_truth}'")
    print(f"   • Dataset audio: {args.ground_truth / 'audio'}")
    print(f"   • Model: {args.model_dir}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Max epochs: {args.max_epochs}")
    print(f"   • Learning rate: {args.learning_rate}")
    for split in SPLITS:
        print(f"   • {split}: {len(load_lst(args.ground_truth / 'lists' / SPLIT_FILES[split][0]))} record(s)")

    checkpoint_path = run_training(
        output_dir=args.output_dir,
        protocol=protocol,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        device=args.device,
        save_top_k=args.save_top_k,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
    )

    print("✅ Training completed.")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_DATABASE_NAME = "MixDataset"
DEFAULT_PROTOCOL_NAME = "MixDiarization"
DEFAULT_MODEL_DIR = Path("models/pyannote-segmentation-3.0")
DEFAULT_CHUNK_DURATION = 10.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_EPOCHS = 1
DEFAULT_MAX_SPEAKERS_PER_CHUNK = 3
DEFAULT_MAX_SPEAKERS_PER_FRAME = 2

SPLITS = ("train", "dev", "test")
DATASET_DIRS = ("audio", "rttm", "uem", "lists")
SPLIT_FILES = {
    "train": {"list": "train.lst", "rttm": "train.rttm", "uem": "train.uem"},
    "dev": {"list": "dev.lst", "rttm": "dev.rttm", "uem": "dev.uem"},
    "test": {"list": "test.lst", "rttm": "test.rttm", "uem": "test.uem"},
}

def collect_dataset_info(dataset_root: Path) -> tuple[dict[str, Path], dict[str, dict[str, Path]], dict[str, list[str]]]:
    dirs = {name: dataset_root / name for name in DATASET_DIRS}
    for label, path in dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} directory under {dataset_root}.")

    split_paths: dict[str, dict[str, Path]] = {}
    split_uris: dict[str, list[str]] = {}
    for split in SPLITS:
        paths = {
            "list": dirs["lists"] / SPLIT_FILES[split]["list"],
            "rttm": dirs["rttm"] / SPLIT_FILES[split]["rttm"],
            "uem": dirs["uem"] / SPLIT_FILES[split]["uem"],
        }
        for kind, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {kind} file for split '{split}': {path}")

        uris = [line.strip() for line in paths["list"].read_text(encoding="utf-8").splitlines() if line.strip()]
        if not uris:
            raise RuntimeError(f"No URIs found in {paths['list']}.")

        split_paths[split] = paths
        split_uris[split] = uris

    for uri in sorted({uri for uris in split_uris.values() for uri in uris}):
        audio_path = dirs["audio"] / f"{uri}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file for URI '{uri}': {audio_path}")

    train_set = set(split_uris["train"])
    dev_set = set(split_uris["dev"])
    test_set = set(split_uris["test"])
    if train_set & dev_set:
        raise RuntimeError("train/dev overlap detected.")
    if train_set & test_set:
        raise RuntimeError("train/test overlap detected.")
    if dev_set & test_set:
        raise RuntimeError("dev/test overlap detected.")

    return dirs, split_paths, split_uris


def write_training_database_yml(
    output_dir: Path,
    audio_dir: Path,
    split_paths: dict[str, dict[str, Path]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    database_yml = output_dir / "database.yml"
    database_yml.write_text(
        "\n".join(
            [
                "Databases:",
                f"  {DEFAULT_DATABASE_NAME}:",
                f"    - {audio_dir.resolve()}/{{uri}}.wav",
                "Protocols:",
                f"  {DEFAULT_DATABASE_NAME}:",
                "    SpeakerDiarization:",
                f"      {DEFAULT_PROTOCOL_NAME}:",
                "        scope: file",
                "        train:",
                f"          uri: {split_paths['train']['list'].resolve()}",
                f"          annotation: {split_paths['train']['rttm'].resolve()}",
                f"          annotated: {split_paths['train']['uem'].resolve()}",
                "        development:",
                f"          uri: {split_paths['dev']['list'].resolve()}",
                f"          annotation: {split_paths['dev']['rttm'].resolve()}",
                f"          annotated: {split_paths['dev']['uem'].resolve()}",
                "        test:",
                f"          uri: {split_paths['test']['list'].resolve()}",
                f"          annotation: {split_paths['test']['rttm'].resolve()}",
                f"          annotated: {split_paths['test']['uem'].resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return database_yml


def load_pretrained_model(model_dir: Path, task) -> Any:
    from pyannote.audio import Model

    model = Model.from_pretrained(model_dir)
    model.task = task
    return model


def train_model(
    output_dir: Path,
    database_yml: Path,
    model_dir: Path,
    batch_size: int,
    max_epochs: int,
    device: str,
) -> Path:
    from pyannote.database import registry
    from pyannote.audio.tasks import SpeakerDiarization
    from pytorch_lightning.callbacks import ModelCheckpoint
    import pytorch_lightning as pl

    registry.load_database(str(database_yml))
    protocol = registry.get_protocol(
        f"{DEFAULT_DATABASE_NAME}.SpeakerDiarization.{DEFAULT_PROTOCOL_NAME}"
    )

    task = SpeakerDiarization(
        protocol,
        duration=DEFAULT_CHUNK_DURATION,
        batch_size=batch_size,
        max_speakers_per_chunk=DEFAULT_MAX_SPEAKERS_PER_CHUNK,
        max_speakers_per_frame=DEFAULT_MAX_SPEAKERS_PER_FRAME,
    )

    model = load_pretrained_model(model_dir, task)
    if model.task is not task:
        raise RuntimeError("Task was not attached to model correctly.")
    print("Model specifications:")
    print(model.specifications)
    print("Task specifications:")
    print(task.specifications)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        dirpath=str(checkpoint_dir),
    )
    trainer = pl.Trainer(
        accelerator="auto" if device == "auto" else device,
        devices=1,
        max_epochs=max_epochs,
        default_root_dir=str(output_dir),
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )
    trainer.fit(model)

    last_checkpoint = getattr(checkpoint_callback, "last_model_path", "")
    if not last_checkpoint:
        raise RuntimeError("Training finished but no checkpoint was saved.")

    print(f"Last checkpoint: {last_checkpoint}")
    return Path(last_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pyannote segmentation-3.0 from an existing dataset root.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Local model directory not found: {args.model_dir}")

    dirs, split_paths, split_uris = collect_dataset_info(args.ground_truth)
    database_yml = write_training_database_yml(args.output_dir, dirs["audio"], split_paths)

    print(f"Ground truth : {args.ground_truth}")
    print(f"Database YML : {database_yml}")
    for split in SPLITS:
        print(f"{split}: {len(split_uris[split])} record(s)")

    checkpoint_path = train_model(
        output_dir=args.output_dir,
        database_yml=database_yml,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        device=args.device,
    )

    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

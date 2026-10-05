#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from pyannote.database.protocol.speaker_diarization import SpeakerDiarizationProtocol
from pyannote.database.util import load_lst, load_rttm, load_uem


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "osd" / "models" / "pyannote-segmentation-3.0"
DEFAULT_CHUNK_DURATION = 10.0 # the value when pretrained, highly recommended not to change
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_EPOCHS = 1
DEFAULT_SAVE_TOP_K = 5
DEFAULT_EARLY_STOPPING_PATIENCE = 0
DEFAULT_LEARNING_RATE = 1e-4
MAX_SPEAKERS_PER_RECORDING = 3
MAX_SPEAKERS_PER_FRAME = 2

SPLITS = ("train", "dev", "test")
SPLIT_FILES = {
    "train": ("train.lst", "train.rttm", "train.uem"),
    "dev": ("dev.lst", "dev.rttm", "dev.uem"),
    "test": ("test.lst", "test.rttm", "test.uem"),
}


def resolve_model_checkpoint(path: Path) -> Path:
    """Resolve either a checkpoint file or a local Hugging Face model directory."""
    if path.is_file():
        return path
    candidate = path / "pytorch_model.bin"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Pretrained segmentation model not found: {path}. Expected a checkpoint "
        "file or a directory containing pytorch_model.bin."
    )


def validate_dataset_layout(dataset_root: Path) -> None:
    """Validate the NvvMix directory structure before parsing annotations."""
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {dataset_root}")
    for split, filenames in SPLIT_FILES.items():
        for directory, filename in zip(("lists", "rttm", "uem"), filenames):
            path = dataset_root / directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing {split} dataset file: {path}")


def build_protocol(dataset_root: Path) -> SpeakerDiarizationProtocol:
    """Build a local pyannote protocol from NvvMix list, RTTM, and UEM files."""
    validate_dataset_layout(dataset_root)

    class LocalDiarizationProtocol(SpeakerDiarizationProtocol):
        def __init__(self) -> None:
            super().__init__()
            self.name = "NvvMix.SpeakerDiarization.local"
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
                if uri not in items["annotations"]:
                    raise ValueError(f"No {split} RTTM annotation found for '{uri}'.")
                if uri not in items["annotated"]:
                    raise ValueError(f"No {split} UEM region found for '{uri}'.")
                audio_path = dataset_root / "audio" / f"{uri}.wav"
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"No {split} audio found for '{uri}': {audio_path}"
                    )
                yield {
                    "uri": uri,
                    "database": "NvvMix",
                    "subset": "development" if split == "dev" else split,
                    "scope": "file",
                    "audio": str(audio_path),
                    "annotation": items["annotations"][uri],
                    "annotated": items["annotated"][uri],
                }

    return LocalDiarizationProtocol()


def maximum_simultaneous_speakers(annotation: Any) -> tuple[int, float | None]:
    """Return peak active unique speakers and the first time that peak occurs."""
    events: dict[float, dict[str, set[str]]] = defaultdict(
        lambda: {"start": set(), "end": set()}
    )
    for label in annotation.labels():
        for segment in annotation.label_timeline(label).support():
            if segment.end <= segment.start:
                continue
            speaker = str(label)
            events[float(segment.start)]["start"].add(speaker)
            events[float(segment.end)]["end"].add(speaker)

    active: set[str] = set()
    peak = 0
    peak_time: float | None = None
    for timestamp in sorted(events):
        active.difference_update(events[timestamp]["end"])
        active.update(events[timestamp]["start"])
        if len(active) > peak:
            peak = len(active)
            peak_time = timestamp
    return peak, peak_time


def validate_protocol(protocol: SpeakerDiarizationProtocol) -> dict[str, int]:
    """Validate split integrity and the native segmentation-3.0 limits."""
    split_counts: dict[str, int] = {}
    uri_splits: dict[str, str] = {}

    iterators = {
        "train": protocol.train,
        "dev": protocol.development,
        "test": protocol.test,
    }
    for split, iterator in iterators.items():
        count = 0
        for file in iterator():
            count += 1
            uri = str(file["uri"])
            if uri in uri_splits:
                raise ValueError(
                    f"Recording '{uri}' appears in both {uri_splits[uri]} and {split}."
                )
            uri_splits[uri] = split

            annotation = file["annotation"]
            labels = [str(label) for label in annotation.labels()]
            if len(labels) > MAX_SPEAKERS_PER_RECORDING:
                raise ValueError(
                    f"Recording '{uri}' in {split} has {len(labels)} unique speakers; "
                    f"the maximum is {MAX_SPEAKERS_PER_RECORDING}: {labels}"
                )

            simultaneous, timestamp = maximum_simultaneous_speakers(annotation)
            if simultaneous > MAX_SPEAKERS_PER_FRAME:
                raise ValueError(
                    f"Recording '{uri}' in {split} has {simultaneous} simultaneously "
                    f"active speakers at {timestamp:.6f}s; the maximum is "
                    f"{MAX_SPEAKERS_PER_FRAME}."
                )

        if count == 0:
            raise ValueError(f"Dataset split '{split}' is empty.")
        split_counts[split] = count

    return split_counts


def validate_native_segmentation(model: Any) -> None:
    """Ensure the pretrained model has the unmodified 3-speaker powerset head."""
    specifications = model.specifications
    if isinstance(specifications, tuple):
        raise ValueError("Expected one segmentation specification, got a multi-task model.")
    if not specifications.powerset:
        raise ValueError("Pretrained segmentation model does not use powerset output.")
    if len(specifications.classes) != MAX_SPEAKERS_PER_RECORDING:
        raise ValueError(
            f"Pretrained model has {len(specifications.classes)} speaker classes; "
            f"expected {MAX_SPEAKERS_PER_RECORDING}."
        )
    if specifications.powerset_max_classes != MAX_SPEAKERS_PER_FRAME:
        raise ValueError(
            "Pretrained model powerset allows "
            f"{specifications.powerset_max_classes} simultaneous classes; expected "
            f"{MAX_SPEAKERS_PER_FRAME}."
        )


def resolve_accelerator(device: str) -> str:
    """Resolve a Lightning accelerator, preferring MPS then CPU for auto."""
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS requested but not available, falling back to CPU.")
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU.")
        return "cpu"
    return device


def count_parameters(model: torch.nn.Module) -> tuple[int, int, list[str]]:
    """Return total/trainable counts and frozen top-level module names."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = sorted(
        {
            name.split(".", 1)[0]
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
    )
    return total, trainable, frozen


def construct_training(
    protocol: SpeakerDiarizationProtocol,
    model_path: Path,
    batch_size: int,
    learning_rate: float,
):
    """Load pretrained segmentation and attach the native diarization task."""
    from pyannote.audio import Model
    from pyannote.audio.tasks import SpeakerDiarization

    checkpoint = resolve_model_checkpoint(model_path)
    model = Model.from_pretrained(checkpoint, map_location="cpu")
    if model is None:
        raise RuntimeError(f"Could not load pretrained segmentation model: {checkpoint}")
    validate_native_segmentation(model)

    model.configure_optimizers = lambda: torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )
    task = SpeakerDiarization(
        protocol,
        duration=DEFAULT_CHUNK_DURATION,
        batch_size=batch_size,
        max_speakers_per_chunk=MAX_SPEAKERS_PER_RECORDING,
        max_speakers_per_frame=MAX_SPEAKERS_PER_FRAME,
    )
    model.task = task
    return model, task


def run_training(
    output_dir: Path,
    protocol: SpeakerDiarizationProtocol,
    model_path: Path,
    batch_size: int,
    max_epochs: int,
    device: str,
    save_top_k: int,
    early_stopping_patience: int,
    learning_rate: float,
    resume: Path | None,
) -> Path | None:
    """Construct, validate, and optionally fine-tune segmentation-3.0."""
    import pyannote.audio
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

    model, task = construct_training(
        protocol=protocol,
        model_path=model_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    task.setup()
    model.setup(stage="fit")
    validate_native_segmentation(model)

    total, trainable, frozen = count_parameters(model)
    print(f"   • Frozen modules: {', '.join(frozen) if frozen else 'none'}")
    print(
        "   • Trainable modules: all"
        if trainable == total
        else "   • Trainable modules: partial"
    )
    print(f"   • Parameters: {total:,} total, {trainable:,} trainable")

    if resume is not None and not resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")

    monitor, mode = task.val_monitor
    checkpoint_kwargs: dict[str, Any] = {
        "dirpath": str(output_dir / "checkpoints"),
        "save_last": True,
        "save_top_k": save_top_k,
    }
    if save_top_k > 0 or early_stopping_patience > 0:
        checkpoint_kwargs.update({"monitor": monitor, "mode": mode})

    class ExperimentMetadata(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
            checkpoint["nvv_mix_experiment"] = {
                "base_model": str(model_path),
                "max_speakers_per_chunk": MAX_SPEAKERS_PER_RECORDING,
                "max_speakers_per_frame": MAX_SPEAKERS_PER_FRAME,
                "chunk_duration": DEFAULT_CHUNK_DURATION,
                "pyannote_audio": pyannote.audio.__version__,
            }

    checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
    callbacks: list[Callback] = [checkpoint_callback, ExperimentMetadata()]
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitor, mode=mode, patience=early_stopping_patience
            )
        )

    trainer = pl.Trainer(
        accelerator=resolve_accelerator(device),
        devices=1,
        max_epochs=max_epochs,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        num_sanity_val_steps=2,
    )
    try:
        trainer.fit(model, ckpt_path=str(resume) if resume else None)
    except RuntimeError as error:
        if resolve_accelerator(device) != "mps":
            raise
        raise RuntimeError(
            "MPS training failed. Retry with --device cpu, or enable PyTorch MPS "
            "CPU fallback when appropriate with PYTORCH_ENABLE_MPS_FALLBACK=1. "
            f"Original error: {error}"
        ) from error

    last_checkpoint = checkpoint_callback.last_model_path
    if not last_checkpoint:
        raise RuntimeError("Training finished but no checkpoint was saved.")
    return Path(last_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pyannote segmentation-3.0 for NvvMix speaker diarization with the native 3-speaker, 2-simultaneous-speaker powerset task.")
    parser.add_argument("--ground-truth",type=Path,required=True,help="NvvMix root containing audio/, lists/, rttm/, and uem/.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Experiment output directory.")
    parser.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL_DIR,help="Pretrained segmentation-3.0 checkpoint or local model directory.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--save-top-k", type=int, default=DEFAULT_SAVE_TOP_K)
    parser.add_argument("--early-stopping-patience",type=int,default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--resume", type=Path, help="Lightning checkpoint from a previous run.")
    parser.add_argument("--device",choices=["auto", "cpu", "mps", "cuda"],default="auto",help="Training device. auto prefers MPS, then CPU.")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate numeric and mutually exclusive training arguments."""
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.save_top_k < -1:
        raise ValueError("--save-top-k must be -1 or greater.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative.")


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    protocol = build_protocol(args.ground_truth)
    split_counts = validate_protocol(protocol)

    print(f"🚀 Train in '{args.ground_truth}'")
    print(f"   • Dataset audio: {args.ground_truth / 'audio'}")
    print(f"   • Model: {args.model_dir}")
    print(f"   • Output: {args.output_dir}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Max epochs: {args.max_epochs}")
    print(f"   • Learning rate: {args.learning_rate}")
    print(f"   • Device: {resolve_accelerator(args.device)}")
    for split in SPLITS:
        print(f"   • {split}: {split_counts[split]} record(s)")
    if args.resume:
        print(f"   • Resume: {args.resume}")

    checkpoint_path = run_training(
        output_dir=args.output_dir,
        protocol=protocol,
        model_path=args.model_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        device=args.device,
        save_top_k=args.save_top_k,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
        resume=args.resume,
    )

    if checkpoint_path is not None:
        print("✅ Training completed.")
        print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

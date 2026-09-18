#!/usr/bin/env python3
"""One-time segmentation-3.0 to native OSDC checkpoint conversion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import Tensor

from .model import NUM_CLASSES, Model, save_native_checkpoint

DEFAULT_SEED = 42


@dataclass(frozen=True)
class TransferReport:
    transferred: tuple[str, ...]
    transformed: tuple[str, ...]
    initialized: tuple[str, ...]
    dropped: tuple[str, ...]


def _model_config(source) -> dict[str, object]:
    """Extract only architecture fields; never copy source task semantics."""
    return {
        "sample_rate": int(source.hparams.sample_rate),
        "num_channels": int(source.hparams.num_channels),
        "sincnet": dict(source.hparams.sincnet),
        "lstm": dict(source.hparams.lstm),
        "linear": dict(source.hparams.linear),
        "num_classes": NUM_CLASSES,
    }


def _mean_rows(value: Tensor, rows: tuple[int, ...]) -> Tensor:
    return value[list(rows)].mean(dim=0)


def _convert_model(source) -> tuple[Model, TransferReport]:
    """Transfer representation and count-compatible parts of the old head."""
    model = Model(model_config=_model_config(source))
    source_state, target_state = source.state_dict(), model.state_dict()
    transferred: list[str] = []
    dropped: list[str] = []

    for name, value in source_state.items():
        if name.startswith("classifier."):
            continue
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value.detach().clone()
            transferred.append(name)
        else:
            dropped.append(name)

    transformed: list[str] = []
    initialized = [
        f"count_head.weight[3:{NUM_CLASSES}]",
        f"count_head.bias[3:{NUM_CLASSES}]",
    ]
    # segmentation-3.0 ordering: silence, 3 singleton speakers, 3 pairs.
    # Averaging removes speaker identity while preserving count-related evidence.
    for suffix in ("weight", "bias"):
        old = source_state[f"classifier.{suffix}"]
        new = target_state[f"count_head.{suffix}"]
        new[0] = old[0]
        new[1] = _mean_rows(old, (1, 2, 3))
        new[2] = _mean_rows(old, (4, 5, 6))
        transformed.append(f"classifier.{suffix}[0:7] -> count_head.{suffix}[0:3]")

    model.load_state_dict(target_state, strict=True)
    dropped.extend(name for name in ("classifier.weight", "classifier.bias") if name not in dropped)
    return model, TransferReport(tuple(transferred), tuple(transformed), tuple(initialized), tuple(dropped))


def run_conversion(output_dir: Path, model_dir: Path) -> tuple[Path, TransferReport]:
    from pyannote.audio import Model as SegmentationModel

    pl.seed_everything(DEFAULT_SEED, workers=True, verbose=False)
    source = SegmentationModel.from_pretrained(str(model_dir))
    model, report = _convert_model(source)
    return save_native_checkpoint(model, output_dir / "checkpoints" / "last.ckpt"), report


def print_report(report: TransferReport) -> None:
    for title, names in (
        ("Transferred", report.transferred), ("Transformed", report.transformed),
        ("Newly initialized", report.initialized), ("Dropped", report.dropped),
    ):
        print(f"   • {title}: {len(names)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Convert segmentation-3.0 into a native {NUM_CLASSES}-class OSDC v0 checkpoint.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("🚀 Convert segmentation-3.0 to a native OSDC checkpoint")
    print(f"   • Model: {args.model_dir}")
    print(f"   • Output: {args.output_dir}")
    path, report = run_conversion(output_dir=args.output_dir, model_dir=args.model_dir)
    print_report(report)
    print("✅ Conversion completed.")
    print(f"Checkpoint: {path}")


if __name__ == "__main__":
    main()

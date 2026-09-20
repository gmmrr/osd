#!/usr/bin/env python3
"""Evaluate a native OSDC checkpoint with per-class frame metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
OSDC_DIR = str(REPO_ROOT / "osdc")
if OSDC_DIR in sys.path:
    sys.path.remove(OSDC_DIR)
sys.path.insert(0, str(REPO_ROOT))

from osdc.model import CLASS_LABELS, Model, OSDCChunkDataset


@torch.inference_mode()
def evaluate_checkpoint(
    model_path: Path, dataset_root: Path, split: str = "test", chunk_duration: float = 10.0,
    batch_size: int = 8, device: str = "cpu",
) -> dict[str, object]:
    selected = torch.device(device)
    model = Model.from_checkpoint(model_path, map_location=selected).eval().to(selected)
    samples = round(chunk_duration * model.sample_rate)
    center, step, duration = model.frame_geometry()
    dataset = OSDCChunkDataset(
        dataset_root, split, chunk_duration, model.num_frames(samples), sample_rate=model.sample_rate,
        frame_center=center, frame_step=step, frame_duration=duration,
    )
    confusion = torch.zeros(len(CLASS_LABELS), len(CLASS_LABELS), dtype=torch.long)
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits = model(batch["waveform"].to(selected))
        metrics = model.evaluate(logits, batch["target"].to(selected), batch["valid"].to(selected))
        confusion += metrics["confusion_matrix"].cpu()
    tp = confusion.diag().float()
    support, predicted = confusion.sum(1).float(), confusion.sum(0).float()
    precision, recall = tp / predicted.clamp_min(1), tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = support > 0
    return {
        "checkpoint": str(model_path), "split": split, "classes": list(CLASS_LABELS),
        "confusion_matrix": confusion.tolist(),
        "accuracy": float(tp.sum() / confusion.sum().clamp_min(1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            label: {"support": int(support[i]), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate(CLASS_LABELS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--chunk-duration", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.model_path, args.ground_truth, args.split, args.chunk_duration, args.batch_size, args.device)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

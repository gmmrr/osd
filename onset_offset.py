#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyannote.core import Annotation
from pyannote.metrics.detection import DetectionErrorRate, DetectionPrecision, DetectionRecall
from matplotlib.ticker import PercentFormatter

from evaluate import load_ground_truth, load_hypothesis, rttm_overlap
from osd import DEFAULT_MODEL_PATH, DEFAULT_SAMPLE_RATE, detect_file, discover_input_files, load_model, resolve_device

REPO_ROOT = Path(__file__).resolve().parent
WORK_ROOT = REPO_ROOT / "onset_offset"
CSV_PATH = WORK_ROOT / "results.csv"
PLOT_PATH = WORK_ROOT / "heatmaps.png"


def frange(start: float, stop: float, step: float) -> list[float]:
    return [round(x, 3) for x in np.arange(start, stop + 1e-9, step)]


def score_combo(rttm_by_uri, uem_by_uri, uris: list[str], hypothesis_by_uri, tolerance: float) -> dict[str, float | int]:
    precision = DetectionPrecision(collar=tolerance)
    recall = DetectionRecall(collar=tolerance)
    der = DetectionErrorRate(collar=tolerance)

    for uri in uris:
        ref = Annotation(uri=uri)
        for idx, segment in enumerate(rttm_overlap(rttm_by_uri[uri])):
            ref[segment, f"track_{idx}"] = "overlap"
        hyp = Annotation(uri=uri)
        for idx, segment in enumerate(hypothesis_by_uri[uri]):
            hyp[segment, f"track_{idx}"] = "overlap"
        uem = uem_by_uri.get(uri)
        precision(ref, hyp, uem=uem)
        recall(ref, hyp, uem=uem)
        der(ref, hyp, uem=uem)

    p = abs(precision)
    r = abs(recall)
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "der": abs(der),
    }


def write_results_csv(results: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "total", "onset", "offset", "precision", "recall", "f1", "der", "work_dir"],
        )
        writer.writeheader()
        writer.writerows(results)


def plot_results(results: list[dict[str, object]], onsets: list[float], offsets: list[float]) -> None:
    shape = (len(offsets), len(onsets))
    precision_grid = np.full(shape, np.nan, dtype=float)
    recall_grid = np.full(shape, np.nan, dtype=float)
    f1_grid = np.full(shape, np.nan, dtype=float)
    der_grid = np.full(shape, np.nan, dtype=float)
    onset_index = {value: idx for idx, value in enumerate(onsets)}
    offset_index = {value: idx for idx, value in enumerate(offsets)}
    for row in results:
        i = offset_index[float(row["offset"])]
        j = onset_index[float(row["onset"])]
        precision_grid[i, j] = float(row["precision"])
        recall_grid[i, j] = float(row["recall"])
        f1_grid[i, j] = float(row["f1"])
        der_grid[i, j] = float(row["der"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    plots = [
        ("Precision", precision_grid, "Reds"),
        ("Recall", recall_grid, "Reds"),
        ("F1", f1_grid, "Reds"),
        ("DER", der_grid, "Reds_r"),
    ]
    extent = [min(onsets), max(onsets), min(offsets), max(offsets)]
    for ax, (title, grid, cmap) in zip(axes.flat, plots):
        finite = grid[np.isfinite(grid)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-9
        image = ax.imshow(grid, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("onset")
        ax.set_ylabel("offset")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle("Onset / Offset Sweep", fontsize=14)
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep onset/offset for OSD.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--onset-min", type=float, required=True)
    parser.add_argument("--onset-max", type=float, required=True)
    parser.add_argument("--offset-min", type=float, required=True)
    parser.add_argument("--offset-max", type=float, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--min-duration-on", type=float, default=0.0)
    parser.add_argument("--min-duration-off", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rttm_by_uri, uem_by_uri, splits = load_ground_truth(args.ground_truth)
    audio_dir = args.ground_truth / "audio"
    files = discover_input_files(audio_dir)
    detector = load_model(args.model_path, resolve_device(args.device))

    onsets = frange(args.onset_min, args.onset_max, args.step)
    offsets = frange(args.offset_min, args.offset_max, args.step)
    combos = list(product(onsets, offsets))
    uris = [uri for split in ("train", "dev", "test") for uri in splits[split]]

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    total = len(combos)
    total_files = total * len(files)
    completed_files = 0
    print(f"🚀 Sweep onset / offset in '{args.ground_truth}'")
    print(f"   • Files: {len(files)}")
    print(f"   • Combinations: {total}")
    print(f"   • Model: {args.model_path}")
    print(f"   • Progress: |{'░' * 20}|   0.00% (0/{total_files})", end="\r", flush=True)
    for index, (onset, offset) in enumerate(combos, start=1):
        combo_dir = WORK_ROOT / f"onset_{onset:.2f}_offset_{offset:.2f}".replace(".", "p")
        if combo_dir.exists():
            if not args.force:
                raise FileExistsError(f"{combo_dir} already exists. Use --force to overwrite.")
            shutil.rmtree(combo_dir)
        combo_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            detect_file(
                detector=detector,
                input_path=file_path,
                output_dir=combo_dir,
                model_path=args.model_path,
                sample_rate=DEFAULT_SAMPLE_RATE,
                onset=onset,
                offset=offset,
                min_duration_on=args.min_duration_on,
                min_duration_off=args.min_duration_off,
                force=True,
            )
            completed_files += 1
            filled = int(round(20 * completed_files / total_files))
            bar = "█" * filled + "░" * (20 - filled)
            pct = 100.0 * completed_files / total_files
            print(
                f"   • Progress: |{bar}| {pct:6.2f}% ({completed_files}/{total_files}) "
                f"[{index}/{total}]",
                end="\r",
                flush=True,
            )

        hypothesis = load_hypothesis(combo_dir)
        scores = score_combo(rttm_by_uri, uem_by_uri, uris, hypothesis, args.tolerance)
        row = {
            "index": index,
            "total": total,
            "onset": onset,
            "offset": offset,
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "der": scores["der"],
            "work_dir": str(combo_dir),
        }
        results.append(row)

    print()
    results.sort(key=lambda row: (-float(row["f1"]), float(row["der"]), float(row["onset"]), float(row["offset"])))
    write_results_csv(results)
    plot_results(results, onsets, offsets)

    print()
    print("✅ Sweep completed.")
    best = results[0]
    print(f"   • Best: onset={float(best['onset']):.2f} offset={float(best['offset']):.2f} F1={100.0 * float(best['f1']):.2f}% DER={100.0 * float(best['der']):.2f}%")
    print(f"   • CSV: {CSV_PATH}")
    print(f"   • Plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()

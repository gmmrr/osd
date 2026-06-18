#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if "\n" not in text:
        return [json.loads(text)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def pick_record(records: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not records:
        raise ValueError("No records found.")
    if index < 0 or index >= len(records):
        raise IndexError(f"Record index {index} out of range for {len(records)} records.")
    return records[index]


def speaker_colors(speaker_labels: list[str]) -> dict[str, str]:
    palette = plt.get_cmap("tab10")
    return {speaker: palette(i % 10) for i, speaker in enumerate(speaker_labels)}


def draw_mix(ax: plt.Axes, record: dict[str, Any]) -> None:
    sources = record.get("sources", [])
    if not sources:
        raise ValueError("Record has no sources.")

    speaker_labels = record.get("speaker_labels") or sorted({src.get("speaker", "unknown") for src in sources})
    colors = speaker_colors(speaker_labels)

    for idx, src in enumerate(sources):
        start = float(src["mix_start"])
        end = float(src["mix_end"])
        speaker = src.get("speaker", "unknown")
        color = colors.get(speaker, "tab:gray")
        ax.barh(
            y=idx,
            width=end - start,
            left=start,
            height=0.55,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )
        label = f'{src["source"]} | {speaker} | {start:.2f}-{end:.2f}s'
        ax.text(start + (end - start) / 2, idx, label, va="center", ha="center", fontsize=9, color="black")

    for ov in record.get("overlap", []):
        start = float(ov["start"])
        end = float(ov["end"])
        if end > start:
            ax.axvspan(start, end, color="gold", alpha=0.18)
            ax.text(
                (start + end) / 2,
                len(sources) - 0.15,
                f'overlap {ov.get("ratio_actual", 0):.2f}',
                ha="center",
                va="bottom",
                fontsize=8,
                color="darkgoldenrod",
            )

    duration = float(record.get("duration", max(float(src["mix_end"]) for src in sources)))
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.6, len(sources) - 0.4)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([src.get("source", f"s{idx+1}") for idx, src in enumerate(sources)])
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f'{record.get("uri", "mix")} | duration={record.get("duration", duration):.2f}s | '
        f'speakers={record.get("speaker_count", len(speaker_labels))} | '
        f'overlap_speaker={record.get("max_overlap_speaker", "?")}'
    )
    ax.grid(axis="x", alpha=0.25)

    legend_items = [Patch(facecolor=colors[speaker], edgecolor="black", label=speaker) for speaker in speaker_labels]
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", frameon=True, fontsize=9)

def plot_mix(record: dict[str, Any], output: Path | None = None, show: bool = True) -> Path | None:
    fig, ax = plt.subplots(figsize=(14, max(3.5, 0.55 * len(record.get("sources", [])) + 2)))
    draw_mix(ax, record)
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output


def plot_mixes(records: list[dict[str, Any]], show: bool = True) -> None:
    if not records:
        raise ValueError("No records found.")

    fig, axes = plt.subplots(
        nrows=len(records),
        ncols=1,
        figsize=(14, max(3.5, 3.5 * len(records))),
        squeeze=False,
    )
    for ax, record in zip(axes.flatten(), records, strict=False):
        draw_mix(ax, record)

    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize mix metadata produced by mix_metadata.py.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a JSON or JSONL mix metadata file.")
    parser.add_argument("--count", type=int, default=5, help="Number of records to visualize from a JSONL file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional image path to save the figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    if len(records) == 1:
        if args.output is not None:
            plot_mix(records[0], output=args.output, show=True)
            print(f"Wrote: {args.output}")
        else:
            plot_mix(records[0], output=None, show=True)
    else:
        subset = records[: args.count]
        if args.output is not None:
            raise ValueError("--output is only supported for a single record input.")
        plot_mixes(subset, show=True)


if __name__ == "__main__":
    main()

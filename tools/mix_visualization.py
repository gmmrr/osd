#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object per line in {path}")
            records.append(item)
        return records

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON structure in {path}")


def pick_record(records: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not records:
        raise ValueError("No records found.")
    if index < 0 or index >= len(records):
        raise IndexError(f"Record index {index} out of range for {len(records)} records.")
    return records[index]


def speaker_colors(speaker_labels: list[str]) -> dict[str, str]:
    palette = plt.get_cmap("tab10")
    return {speaker: palette(i % 10) for i, speaker in enumerate(speaker_labels)}


def sort_speaker_labels(speaker_labels: list[str]) -> list[str]:
    def sort_key(label: str) -> tuple[int, str]:
        suffix = label.rsplit("spk", 1)[-1]
        try:
            return int(suffix), label
        except ValueError:
            return 10**9, label

    return sorted(speaker_labels, key=sort_key)


def sort_records(records: list[dict[str, Any]], key_name: str, reverse: bool) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: float(record.get(key_name, 0.0)), reverse=reverse)


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def format_overlap_ratio(ov: dict[str, Any]) -> str | None:
    value = ov.get("ratio_actual")
    if value is None:
        value = ov.get("ratio_target")
    if value is None:
        return None
    return f"{float(value):.2f}"


def format_overlap_summary(record: dict[str, Any]) -> str:
    overlaps = record.get("overlap") or []
    center = first_present(
        record.get("overlap_ratio_center"),
        record.get("overlap_ratio_target"),
        overlaps[0].get("ratio_target") if overlaps else None,
        overlaps[0].get("ratio_actual") if overlaps else None,
    )
    offset = record.get("overlap_ratio_offset", 0.0)
    if center is not None:
        return f"overlap={float(center):.2f} ± {float(offset):.2f}"

    return "overlap=?"


def draw_mix(ax: plt.Axes, record: dict[str, Any]) -> None:
    sources = record.get("sources", [])
    if not sources:
        raise ValueError("Record has no sources.")

    speaker_labels = record.get("speaker_labels") or []
    if not speaker_labels:
        speaker_labels = list(dict.fromkeys(src.get("speaker", "unknown") for src in sources))

    speaker_labels = sort_speaker_labels(speaker_labels[:3])
    colors = speaker_colors(speaker_labels)
    speaker_rows = {speaker: idx for idx, speaker in enumerate(speaker_labels)}
    if not speaker_rows:
        raise ValueError("Record has no speaker labels.")

    total_sources = len(sources)

    for src in sources:
        start = float(src["mix_start"])
        end = float(src["mix_end"])
        speaker = src.get("speaker", "unknown")
        if speaker not in speaker_rows:
            continue
        idx = speaker_rows[speaker]
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
        label = f"{start:.2f}-{end:.2f}s"
        ax.text(start + (end - start) / 2, idx, label, va="center", ha="center", fontsize=8, color="black")

    for ov in record.get("overlap", []):
        start = float(ov["start"])
        end = float(ov["end"])
        if end > start:
            ax.axvspan(start, end, color="gold", alpha=0.18)
            label = format_overlap_ratio(ov)
            if label is not None:
                ax.text(
                    (start + end) / 2,
                    -0.38,
                    label,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="darkgoldenrod",
                )

    duration = float(record.get("duration", max(float(src["mix_end"]) for src in sources)))
    ax.set_xlim(0, 25.0)
    ax.set_ylim(-0.6, len(speaker_labels) - 0.4)
    ax.set_yticks(range(len(speaker_labels)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        f'{record.get("uri", "mix")} | duration={record.get("duration", duration):.2f}s | segments={total_sources} | '
        f'speakers={record.get("speaker_count", len(speaker_labels))} | '
        f'max speaker per frame={record.get("max_speakers_per_frame", record.get("max_overlap_speaker", "?"))} | '
        f'{format_overlap_summary(record)}'
    )
    ax.grid(axis="x", alpha=0.25)

    legend_items = [Patch(facecolor=colors[speaker], edgecolor="black", label=speaker) for speaker in speaker_labels]
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", frameon=True, fontsize=9)

def plot_mix(record: dict[str, Any], output: Path | None = None, show: bool = True) -> Path | None:
    speaker_labels = record.get("speaker_labels") or []
    fig, ax = plt.subplots(figsize=(16, max(3.5, 0.9 * max(1, min(3, len(speaker_labels))) + 2)))
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
    parser.add_argument("--count", type=int, default=5, help="Number of records to visualize from the input file.")
    parser.add_argument(
        "--show-max-speaker",
        action="store_true",
        help="Sort records by speaker_count descending and show the top `--count` records.",
    )
    parser.add_argument(
        "--show-max-segment",
        action="store_true",
        help="Sort records by source_count descending and show the top `--count` records.",
    )
    parser.add_argument(
        "--show-min-speaker",
        action="store_true",
        help="Sort records by speaker_count ascending and show the top `--count` records.",
    )
    parser.add_argument(
        "--show-min-segment",
        action="store_true",
        help="Sort records by source_count ascending and show the top `--count` records.",
    )
    parser.add_argument(
        "--show-max-duration",
        action="store_true",
        help="Sort records by duration descending and show the top `--count` records.",
    )
    parser.add_argument(
        "--show-min-duration",
        action="store_true",
        help="Sort records by duration ascending and show the top `--count` records.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional image path to save the figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    for enabled, key_name, reverse in (
        (args.show_max_segment, "source_count", True),
        (args.show_min_segment, "source_count", False),
        (args.show_max_speaker, "speaker_count", True),
        (args.show_min_speaker, "speaker_count", False),
        (args.show_max_duration, "duration", True),
        (args.show_min_duration, "duration", False),
    ):
        if enabled:
            records = sort_records(records, key_name, reverse)
            break

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

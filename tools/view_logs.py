#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path

from tensorboardX.proto import event_pb2


def read_tfrecords(path: Path):
    with path.open("rb") as f:
        while True:
            header = f.read(8)
            if not header:
                return
            if len(header) != 8:
                return
            (length,) = struct.unpack("<Q", header)
            f.read(4)
            data = f.read(length)
            if len(data) != length:
                return
            f.read(4)
            yield data


def collect_scalars(event_file: Path) -> dict[int, list[tuple[str, float]]]:
    scalars: dict[int, list[tuple[str, float]]] = {}
    for raw in read_tfrecords(event_file):
        ev = event_pb2.Event()
        ev.ParseFromString(raw)
        if not ev.summary.value:
            continue
        values: list[tuple[str, float]] = []
        for value in ev.summary.value:
            if value.HasField("simple_value"):
                values.append((value.tag, float(value.simple_value)))
        if values:
            scalars.setdefault(ev.step, []).extend(values)
    return scalars


def short_label(tag: str) -> str:
    tag = tag.replace("DiarizationErrorRate", "DER")
    if len(tag) <= 18:
        return tag
    for sep in ("/", "_", "-", "."):
        if sep in tag:
            return tag.rsplit(sep, 1)[-1]
    return tag[:18]


def print_wide_table(scalars: dict[int, list[tuple[str, float]]]) -> None:
    tags = sorted({tag for values in scalars.values() for tag, _ in values if tag not in {"epoch", "step", "hp_metric"}})
    labels = {tag: short_label(tag) for tag in tags}

    rows: dict[int, dict[str, float]] = {}
    for step in sorted(scalars):
        values = scalars[step]
        values_by_tag = {tag: value for tag, value in values}
        epoch_value = int(values_by_tag.get("epoch", 0))
        row = rows.setdefault(epoch_value, {})
        row["epoch"] = float(epoch_value)
        row["step"] = float(step)
        for tag, value in values:
            if tag not in {"epoch", "step", "hp_metric"}:
                row[tag] = value

    epoch_width = max(len("epoch"), max((len(str(epoch)) for epoch in rows), default=5))
    step_width = max(len("step"), max((len(str(int(values.get("step", 0.0)))) for values in rows.values()), default=4))
    widths = {tag: max(12, len(labels[tag])) for tag in tags}
    columns = [("epoch", epoch_width), ("step", step_width), *[(tag, widths[tag]) for tag in tags]]

    def border(left: str, fill: str, join: str, right: str) -> str:
        parts = [fill * (width + 2) for _, width in columns]
        return left + join.join(parts) + right

    def row(cells: list[str]) -> str:
        return "│ " + " │ ".join(cells[i].ljust(width) for i, (_, width) in enumerate(columns)) + " │"

    print(border("┌", "─", "┬", "┐"))
    print(row(["epoch", "step", *[labels[tag] for tag in tags]]))
    print(border("├", "─", "┼", "┤"))
    for epoch in sorted(rows):
        values = rows[epoch]
        if epoch == 0 and values.get("hp_metric") == -1.0 and set(values) <= {"epoch", "step", "hp_metric"}:
            continue
        cells = [str(epoch), str(int(values.get("step", 0)))]
        for tag in tags:
            value = values.get(tag)
            cells.append(f"{value:.6f}" if value is not None else "")
        print(row(cells))
    print(border("└", "─", "┴", "┘"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path, help="例如 models/segmentation-3.0-ft-train-v3/lightning_logs/version_0")
    args = parser.parse_args()

    event_files = sorted(args.log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event file found in {args.log_dir}")

    scalars = collect_scalars(event_files[0])
    if not scalars:
        print("No scalar values found.")
        return

    print(f"Log dir : {args.log_dir}")
    print(f"Event   : {event_files[0].name}")
    print(f"Tags    : {len({tag for values in scalars.values() for tag, _ in values})}")

    print_wide_table(scalars)


if __name__ == "__main__":
    main()

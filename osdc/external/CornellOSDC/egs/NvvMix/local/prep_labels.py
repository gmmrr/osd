#!/usr/bin/env python3
"""Convert NvvMix RTTM databases to Cornell OSDC label WAVs and manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


SPLITS = ("train", "dev", "test")
HOP_SIZE = Decimal("0.01")
LABEL_SAMPLE_RATE = 16_000
MAX_LABEL = 4


def read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def discover_databases(source_root: Path) -> list[Path]:
    if source_root.is_file():
        return [source_root]
    return sorted(source_root.rglob("database.yml"))


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def source_name(source_root: Path, database_path: Path) -> str:
    source_dir = database_path.parent.resolve()
    try:
        relative = source_dir.relative_to(source_root.resolve())
        parts = relative.parts
    except ValueError:
        parts = (source_dir.name,)
    return "__".join(parts) if parts else source_dir.name


def load_database(source_root: Path, database_path: Path) -> dict:
    config = yaml.safe_load(database_path.read_text(encoding="utf-8"))["dataset"]
    root = Path(config.get("root", "."))
    if not root.is_absolute():
        root = database_path.parent / root
    root = root.resolve()
    return {
        "source": source_name(source_root, database_path),
        "root": root,
        "audio": config["audio"],
        "splits": config["splits"],
    }


def parse_uem(path: Path) -> dict[str, Decimal]:
    durations: dict[str, Decimal] = {}
    for line in read_lines(path):
        fields = line.split()
        uri, start, end = fields[0], Decimal(fields[2]), Decimal(fields[3])
        if start != 0:
            raise ValueError(f"NvvMix UEM must start at zero: {path}: {line}")
        durations[uri] = max(durations.get(uri, Decimal(0)), end)
    return durations


def parse_rttm(path: Path) -> dict[str, dict[str, list[tuple[Decimal, Decimal]]]]:
    activity: dict[str, dict[str, list[tuple[Decimal, Decimal]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for line in read_lines(path):
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            raise ValueError(f"Invalid RTTM line in {path}: {line}")
        uri, speaker = fields[1], fields[7]
        start, duration = Decimal(fields[3]), Decimal(fields[4])
        activity[uri][speaker].append((start, start + duration))
    return activity


def to_frame(value: Decimal, rounding: str) -> int:
    return int((value / HOP_SIZE).to_integral_value(rounding=rounding))


def make_labels(
    duration: Decimal,
    speakers: dict[str, list[tuple[Decimal, Decimal]]],
) -> np.ndarray:
    frame_count = to_frame(duration, ROUND_CEILING)
    difference = np.zeros(frame_count + 1, dtype=np.int16)
    for intervals in speakers.values():
        speaker_mask = np.zeros(frame_count, dtype=bool)
        for start, end in intervals:
            first = max(0, to_frame(start, ROUND_FLOOR))
            last = min(frame_count, to_frame(end, ROUND_FLOOR))
            speaker_mask[first:last] = True
        changes = np.diff(np.pad(speaker_mask.astype(np.int16), (1, 1)))
        difference[:-1] += changes[:-1]
        difference[-1] += changes[-1]
    return np.clip(np.cumsum(difference[:-1]), 0, MAX_LABEL).astype(np.float32)


def convert_database(
    database: dict,
    output_root: Path,
    *,
    force: bool,
    dry_run: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    source_root = output_root / database["source"]

    for split in SPLITS:
        split_config = database["splits"][split]
        list_path = resolve_path(database["root"], split_config["list"])
        rttm_path = resolve_path(database["root"], split_config["rttm"])
        uem_path = resolve_path(database["root"], split_config["uem"])
        uris = read_lines(list_path)
        durations = parse_uem(uem_path)
        activity = parse_rttm(rttm_path)
        if set(uris) != set(durations):
            raise ValueError(f"List/UEM mismatch: {list_path} and {uem_path}")

        records = []
        label_dir = source_root / "labels" / split
        for uri in uris:
            audio_path = resolve_path(
                database["root"], database["audio"].format(uri=uri)
            )
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            label_path = (label_dir / f"LABEL-{uri}.wav").resolve()
            records.append(
                {
                    "source": database["source"],
                    "split": split,
                    "uri": uri,
                    "audio": str(audio_path),
                    "label": str(label_path),
                }
            )
            if not dry_run and (force or not label_path.exists()):
                label_path.parent.mkdir(parents=True, exist_ok=True)
                labels = make_labels(durations[uri], activity.get(uri, {}))
                sf.write(label_path, labels, LABEL_SAMPLE_RATE, subtype="FLOAT")

        manifest_path = source_root / "manifests" / f"{split}.json"
        if not dry_run:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if manifest_path.exists() and not force:
                raise FileExistsError(f"Manifest already exists: {manifest_path}")
            manifest_path.write_text(
                json.dumps(records, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        counts[split] = len(records)

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert each NvvMix database.yml to Cornell OSDC labels without copying audio.")
    parser.add_argument("--input-dir",type=Path,required=True,help="Root containing NvvMix database.yml files.")
    parser.add_argument("--output-dir",type=Path,required=True,help="Directory for generated labels and manifests.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    databases = discover_databases(args.input_dir)
    if not databases:
        raise FileNotFoundError(f"No database.yml found under {args.input_dir}")

    for database_path in databases:
        database = load_database(args.input_dir, database_path)
        counts = convert_database(
            database,
            args.output_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
        summary = ", ".join(f"{split}={counts[split]}" for split in SPLITS)
        print(f"{database['source']}: {summary}")


if __name__ == "__main__":
    main()

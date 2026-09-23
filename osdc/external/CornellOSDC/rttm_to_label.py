#!/usr/bin/env python3
"""Convert repository RTTM/UEM ground truth to Cornell OSDC labels.

``SOURCE_ROOT`` may be one dataset directory or a parent containing several
dataset directories. Every ``database.yml`` below it is discovered. The name
of the directory containing a database (for example, ``train``) has no split
meaning: train/dev/test membership comes exclusively from that database's UEM
files. Lists are used only to validate the UEM membership.

No audio is opened. Each output WAV sample is one 10 ms label, even though the
WAV header uses 16 kHz for compatibility with Cornell OSDC.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import yaml


SPLITS = ("train", "dev", "test")
HOP_SIZE = Decimal("0.01")
LABEL_SAMPLE_RATE = 16_000
MAX_LABEL = 4


@dataclass(frozen=True, order=True)
class Interval:
    """Half-open interval in seconds: [start, end)."""

    start: Decimal
    end: Decimal


@dataclass(frozen=True)
class SplitPaths:
    """Metadata paths declared for one split in database.yml."""

    uem: Path
    rttm: Path
    recording_list: Path | None


@dataclass(frozen=True)
class DatasetSpec:
    """One discovered database and its existing naming convention."""

    database: Path
    audio_template: str
    splits: Mapping[str, SplitPaths]


@dataclass
class SplitStats:
    """Aggregate label statistics for one output split."""

    recordings: int = 0
    frames: int = 0
    class_frames: np.ndarray = field(default_factory=lambda: np.zeros(MAX_LABEL + 1, dtype=np.int64))
    max_raw_speakers: int = 0


@dataclass(frozen=True)
class RecordingPlan:
    """Fully validated input and output mapping for one recording."""

    split: str
    recording_id: str
    regions: Sequence[Interval]
    speakers: Mapping[str, Sequence[Interval]]
    output_path: Path
    source_database: Path


def data_lines(path: Path) -> Iterable[tuple[int, str]]:
    """Yield nonempty, noncomment lines and their one-based line numbers."""

    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    yield line_number, line
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def parse_decimal(value: str, *, path: Path, line_number: int, field_name: str) -> Decimal:
    """Parse a finite Decimal and retain exact decimal timing."""

    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {field_name} at {path}:{line_number}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"Non-finite {field_name} at {path}:{line_number}: {value!r}")
    return result


def discover_databases(source_root: Path) -> list[Path]:
    """Find every database.yml without assigning meaning to directory names."""

    if source_root.is_file():
        if source_root.name != "database.yml":
            raise ValueError(f"Expected database.yml, got file: {source_root}")
        databases = [source_root]
    elif source_root.is_dir():
        direct = source_root / "database.yml"
        databases = [direct] if direct.is_file() else sorted(source_root.rglob("database.yml"))
    else:
        raise ValueError(f"Source path does not exist: {source_root}")
    if not databases:
        raise ValueError(f"No database.yml found under {source_root}")
    return [path.resolve() for path in databases]


def resolve_metadata_path(dataset_root: Path, value: Any, *, field_name: str) -> Path:
    """Resolve one database metadata path relative to dataset.root."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"database.yml field {field_name!r} must be a nonempty path")
    path = Path(value)
    return (path if path.is_absolute() else dataset_root / path).resolve()


def load_database(database_path: Path) -> DatasetSpec:
    """Load the database-provided UEM/RTTM/list paths for all three splits."""

    try:
        config = yaml.safe_load(database_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read {database_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in {database_path}: {exc}") from exc
    if not isinstance(config, Mapping) or not isinstance(config.get("dataset"), Mapping):
        raise ValueError(f"{database_path} must contain a 'dataset' mapping")

    dataset = config["dataset"]
    configured_root = dataset.get("root", ".")
    if not isinstance(configured_root, str):
        raise ValueError(f"dataset.root must be a path string in {database_path}")
    dataset_root = Path(configured_root)
    if not dataset_root.is_absolute():
        dataset_root = database_path.parent / dataset_root
    dataset_root = dataset_root.resolve()

    audio_template = dataset.get("audio")
    if not isinstance(audio_template, str) or "{uri}" not in audio_template:
        raise ValueError(f"dataset.audio must contain '{{uri}}' in {database_path}")
    configured_splits = dataset.get("splits")
    if not isinstance(configured_splits, Mapping):
        raise ValueError(f"{database_path} must contain dataset.splits")

    split_paths: dict[str, SplitPaths] = {}
    for split in SPLITS:
        entry = configured_splits.get(split)
        if not isinstance(entry, Mapping):
            raise ValueError(f"{database_path} does not define split {split!r}")
        list_value = entry.get("list")
        split_paths[split] = SplitPaths(
            uem=resolve_metadata_path(dataset_root, entry.get("uem"), field_name=f"splits.{split}.uem"),
            rttm=resolve_metadata_path(dataset_root, entry.get("rttm"), field_name=f"splits.{split}.rttm"),
            recording_list=(
                resolve_metadata_path(dataset_root, list_value, field_name=f"splits.{split}.list")
                if list_value is not None
                else None
            ),
        )
    return DatasetSpec(database=database_path, audio_template=audio_template, splits=split_paths)


def parse_list(path: Path) -> set[str]:
    """Parse a split list for consistency checking only."""

    recordings: set[str] = set()
    for line_number, line in data_lines(path):
        fields = line.split()
        if len(fields) != 1:
            raise ValueError(f"Malformed list line at {path}:{line_number}: {line!r}")
        if fields[0] in recordings:
            raise ValueError(f"Duplicate recording ID {fields[0]!r} at {path}:{line_number}")
        recordings.add(fields[0])
    if not recordings:
        raise ValueError(f"Split list is empty: {path}")
    return recordings


def merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    """Return the union of overlapping or touching half-open intervals."""

    merged: list[Interval] = []
    for interval in sorted(intervals):
        if not merged or interval.start > merged[-1].end:
            merged.append(interval)
        else:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, interval.end))
    return merged


def parse_uem(path: Path) -> dict[str, list[Interval]]:
    """Parse UEM; its recording IDs are the authoritative split membership."""

    regions: dict[str, list[Interval]] = defaultdict(list)
    seen: set[tuple[str, Decimal, Decimal]] = set()
    for line_number, line in data_lines(path):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed UEM line at {path}:{line_number}: {line!r}")
        recording_id = fields[0]
        start = parse_decimal(fields[2], path=path, line_number=line_number, field_name="UEM start")
        end = parse_decimal(fields[3], path=path, line_number=line_number, field_name="UEM end")
        if start < 0:
            raise ValueError(f"Negative UEM start at {path}:{line_number}: {start}")
        if end <= start:
            raise ValueError(f"UEM end must exceed start at {path}:{line_number}: {start}..{end}")
        key = (recording_id, start, end)
        if key in seen:
            raise ValueError(f"Duplicate UEM region at {path}:{line_number}: {recording_id} {start}..{end}")
        seen.add(key)
        regions[recording_id].append(Interval(start, end))
    if not regions:
        raise ValueError(f"UEM file is empty: {path}")
    normalized: dict[str, list[Interval]] = {}
    for recording_id, items in regions.items():
        ordered = sorted(items)
        for previous, current in zip(ordered, ordered[1:]):
            if current.start < previous.end:
                raise ValueError(
                    f"Overlapping UEM regions for {recording_id!r} in {path}: "
                    f"{previous.start}..{previous.end} and {current.start}..{current.end}"
                )
        normalized[recording_id] = merge_intervals(ordered)
    return normalized


def segment_within_uem(segment: Interval, regions: Sequence[Interval]) -> bool:
    """Return whether an RTTM segment is inside the union of UEM regions."""

    return any(region.start <= segment.start and segment.end <= region.end for region in regions)


def parse_rttm(
    path: Path,
    uem: Mapping[str, Sequence[Interval]],
) -> dict[str, dict[str, list[Interval]]]:
    """Parse RTTM activity while retaining speaker identity for de-duplication."""

    expected_ids = set(uem)
    activity: dict[str, dict[str, list[Interval]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, Decimal, Decimal, str]] = set()
    duplicate_count = 0
    for line_number, line in data_lines(path):
        fields = line.split()
        if len(fields) < 8 or fields[0].upper() != "SPEAKER":
            raise ValueError(f"Malformed RTTM SPEAKER line at {path}:{line_number}: {line!r}")
        recording_id = fields[1]
        if recording_id not in expected_ids:
            raise ValueError(f"Unexpected RTTM recording ID {recording_id!r} at {path}:{line_number}")
        start = parse_decimal(fields[3], path=path, line_number=line_number, field_name="RTTM start")
        duration = parse_decimal(fields[4], path=path, line_number=line_number, field_name="RTTM duration")
        speaker_id = fields[7]
        if start < 0:
            raise ValueError(f"Negative RTTM start at {path}:{line_number}: {start}")
        if duration <= 0:
            raise ValueError(f"RTTM duration must be positive at {path}:{line_number}: {duration}")
        if not speaker_id or speaker_id == "<NA>":
            raise ValueError(f"Missing RTTM speaker ID at {path}:{line_number}")
        segment = Interval(start, start + duration)
        if not segment_within_uem(segment, uem[recording_id]):
            raise ValueError(
                f"RTTM segment lies outside UEM for {recording_id!r} at {path}:{line_number}: "
                f"{segment.start}..{segment.end}"
            )
        key = (recording_id, segment.start, segment.end, speaker_id)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        activity[recording_id][speaker_id].append(segment)
    if duplicate_count:
        print(f"warning: ignored {duplicate_count} exact duplicate RTTM entries in {path}", file=sys.stderr)
    return {recording_id: dict(speakers) for recording_id, speakers in activity.items()}


def floor_frame(time: Decimal) -> int:
    """Map seconds to a frame index using Cornell's floor policy."""

    return int((time / HOP_SIZE).to_integral_value(rounding=ROUND_FLOOR))


def timeline_frames(time: Decimal) -> int:
    """Use ceil for UEM duration so the timeline covers its final partial frame."""

    return int((time / HOP_SIZE).to_integral_value(rounding=ROUND_CEILING))


def rasterize_labels(
    recording_id: str,
    regions: Sequence[Interval],
    speakers: Mapping[str, Sequence[Interval]],
) -> tuple[np.ndarray, int]:
    """Produce 100 Hz 0/1/2/3/4+ labels and the unclipped maximum count.

    Decimal arithmetic prevents floating-point accumulation. RTTM intervals
    are half-open and use the same frame policy as Cornell ``prep_labels.py``:
    ``floor(start / 0.01):floor(end / 0.01)``. The UEM timeline uses ceil at
    its final end so trailing silence and a partial final frame are retained.
    Intervals belonging to the same speaker are unioned before counting.
    """

    frame_count = timeline_frames(max(region.end for region in regions))
    if frame_count <= 0:
        raise ValueError(f"UEM timeline for {recording_id!r} has no frames")
    difference = np.zeros(frame_count + 1, dtype=np.int32)

    for intervals in speakers.values():
        speaker_frames = [
            Interval(Decimal(floor_frame(segment.start)), Decimal(floor_frame(segment.end)))
            for segment in intervals
        ]
        for interval in merge_intervals(speaker_frames):
            start = max(0, int(interval.start))
            end = min(frame_count, int(interval.end))
            if end > start:
                difference[start] += 1
                difference[end] -= 1

    raw_counts = np.cumsum(difference[:-1], dtype=np.int32)
    max_raw = int(raw_counts.max(initial=0))
    labels = np.minimum(raw_counts, MAX_LABEL).astype(np.int16, copy=False)
    if labels.ndim != 1:
        raise AssertionError(f"Labels for {recording_id!r} are not one-dimensional")
    if len(labels) != frame_count:
        raise AssertionError(f"Label length mismatch for {recording_id!r}: {len(labels)} != {frame_count}")
    if not np.isfinite(labels).all():
        raise AssertionError(f"Labels for {recording_id!r} contain NaN or infinity")
    if labels.min(initial=0) < 0 or labels.max(initial=0) > MAX_LABEL:
        raise AssertionError(f"Labels for {recording_id!r} fall outside 0..{MAX_LABEL}")
    return labels, max_raw


def derive_label_filename(audio_template: str, recording_id: str) -> str:
    """Reuse the existing database audio basename as Cornell's label identity."""

    try:
        rendered = audio_template.format(uri=recording_id)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Cannot apply audio template {audio_template!r}: {exc}") from exc
    filename = Path(rendered).name
    if Path(filename).suffix.lower() != ".wav":
        raise ValueError(f"dataset.audio does not produce a WAV filename: {rendered!r}")
    cornell_id = Path(filename).stem.split("-")[-1]
    if cornell_id != recording_id:
        raise ValueError(
            f"Label filename {filename!r} maps to {cornell_id!r} in Cornell OnlineFeats, "
            f"not recording {recording_id!r}"
        )
    return filename


def validate_list(path: Path | None, uem_ids: set[str]) -> None:
    """Require a declared list to agree exactly with UEM."""

    if path is None:
        return
    listed = parse_list(path)
    if listed != uem_ids:
        only_uem = sorted(uem_ids - listed)
        only_list = sorted(listed - uem_ids)
        raise ValueError(
            f"List/UEM mismatch for {path}: {len(only_uem)} only in UEM, {len(only_list)} only in list; "
            f"examples UEM={only_uem[:3]}, list={only_list[:3]}"
        )


def build_plan(databases: Sequence[DatasetSpec], output_root: Path) -> list[RecordingPlan]:
    """Validate every source and construct a collision-free global output plan."""

    plans: list[RecordingPlan] = []
    recording_owner: dict[str, tuple[str, Path]] = {}
    output_owner: dict[Path, tuple[str, Path]] = {}

    for dataset in databases:
        print(f"Database: {dataset.database}")
        for split in SPLITS:
            paths = dataset.splits[split]
            uem = parse_uem(paths.uem)
            validate_list(paths.recording_list, set(uem))
            activity = parse_rttm(paths.rttm, uem)
            print(f"  {split}: {len(uem):,} recordings from {paths.uem}")

            for recording_id, regions in uem.items():
                previous = recording_owner.get(recording_id)
                if previous is not None:
                    raise ValueError(
                        f"Recording {recording_id!r} occurs more than once: {previous[0]} in {previous[1]} "
                        f"and {split} in {dataset.database}"
                    )
                filename = derive_label_filename(dataset.audio_template, recording_id)
                output_path = (output_root / split / filename).resolve()
                previous_output = output_owner.get(output_path)
                if previous_output is not None:
                    raise ValueError(
                        f"Output collision at {output_path}: {previous_output[0]!r} from {previous_output[1]} "
                        f"and {recording_id!r} from {dataset.database}"
                    )
                recording_owner[recording_id] = (split, dataset.database)
                output_owner[output_path] = (recording_id, dataset.database)
                plans.append(
                    RecordingPlan(
                        split=split,
                        recording_id=recording_id,
                        regions=regions,
                        speakers=activity.get(recording_id, {}),
                        output_path=output_path,
                        source_database=dataset.database,
                    )
                )
    return plans


def execute_plan(
    plans: Sequence[RecordingPlan],
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, SplitStats]:
    """Rasterize labels, optionally write them, and collect split statistics."""

    if not dry_run and not overwrite:
        existing = [plan.output_path for plan in plans if plan.output_path.exists()]
        if existing:
            raise FileExistsError(f"{len(existing)} output files already exist; first: {existing[0]} (use --overwrite)")
    if not dry_run:
        for split in SPLITS:
            split_plans = [plan for plan in plans if plan.split == split]
            if split_plans:
                split_plans[0].output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {split: SplitStats() for split in SPLITS}
    for plan in plans:
        labels, max_raw = rasterize_labels(plan.recording_id, plan.regions, plan.speakers)
        split_stats = stats[plan.split]
        split_stats.recordings += 1
        split_stats.frames += len(labels)
        split_stats.class_frames += np.bincount(labels, minlength=MAX_LABEL + 1)
        split_stats.max_raw_speakers = max(split_stats.max_raw_speakers, max_raw)
        if not dry_run:
            sf.write(plan.output_path, labels.astype(np.float32), LABEL_SAMPLE_RATE, subtype="FLOAT")
    return stats


def print_summary(stats: Mapping[str, SplitStats], *, dry_run: bool) -> None:
    """Print recording counts, duration, class balance, and raw overlap maxima."""

    print("\nSummary" + (" (dry run; no files written)" if dry_run else ""))
    for split in SPLITS:
        item = stats[split]
        print(f"\n{split}")
        print(f"  recordings: {item.recordings:,}")
        print(f"  total frames: {item.frames:,}")
        print(f"  duration represented: {item.frames / 100:.2f} s ({item.frames / 360_000:.2f} h)")
        for label, count in enumerate(item.class_frames):
            name = "4+" if label == MAX_LABEL else str(label)
            percentage = 100.0 * int(count) / item.frames if item.frames else 0.0
            print(f"  class {name}: {int(count):,} ({percentage:.4f}%)")
        print(f"  max raw speakers: {item.max_raw_speakers}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively discover database.yml files and convert their UEM-defined train/dev/test RTTM ground "
            "truth to 100 Hz Cornell OSDC label WAVs."
        )
    )
    parser.add_argument("source_root", type=Path, help="Dataset directory, database.yml, or parent of dataset directories")
    parser.add_argument("output_root", type=Path, help="Output root containing merged train/dev/test label directories")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print statistics without writing labels")
    parser.add_argument("--overwrite", action="store_true", help="Replace output label WAV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_paths = discover_databases(args.source_root)
    databases = [load_database(path) for path in database_paths]
    print(f"Discovered {len(databases)} database(s) under {args.source_root}")
    print("Split source of truth: UEM paths declared by each database.yml")
    plans = build_plan(databases, args.output_root)
    print(f"Output naming: dataset.audio basename; example {plans[0].output_path.name!r}")
    stats = execute_plan(plans, dry_run=args.dry_run, overwrite=args.overwrite)
    print_summary(stats, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileExistsError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

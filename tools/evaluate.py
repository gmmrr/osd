#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm, load_uem
from pyannote.metrics.detection import DetectionErrorRate, DetectionPrecision, DetectionRecall


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}: {type(data).__name__}")


def timeline_from_intervals(uri: str, intervals: list[dict[str, Any]]) -> Timeline:
    timeline = Timeline(uri=uri)
    for item in intervals:
        try:
            start = float(item["start"])
            end = float(item["end"])
        except Exception:
            continue
        if end > start:
            timeline.add(Segment(start, end))
    return timeline.support()


def timeline_to_annotation(timeline: Timeline, label: str = "overlap") -> Annotation:
    annotation = Annotation(uri=timeline.uri)
    for idx, segment in enumerate(timeline):
        annotation[segment, f"track_{idx}"] = label
    return annotation


def annotation_to_overlap_timeline(annotation: Annotation) -> Timeline:
    overlap = Timeline(uri=annotation.uri)
    boundaries = set()
    for segment, _, _ in annotation.itertracks(yield_label=True):
        boundaries.add(segment.start)
        boundaries.add(segment.end)
    boundaries = sorted(boundaries)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        middle = 0.5 * (start + end)
        active_speakers = set()
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            if segment.start <= middle < segment.end:
                active_speakers.add(speaker)
        if len(active_speakers) >= 2:
            overlap.add(Segment(start, end))
    return overlap.support()


def load_rttm_annotations(rttm_path: Path) -> dict[str, Annotation]:
    paths = sorted(rttm_path.rglob("*.rttm")) if rttm_path.is_dir() else [rttm_path]
    annotations: dict[str, Annotation] = {}
    for path in paths:
        loaded = load_rttm(path)
        if not isinstance(loaded, dict):
            raise TypeError("Expected load_rttm to return a dictionary {uri: Annotation}.")
        annotations.update(loaded)
    return annotations


def load_split_uris(lists_dir: Path) -> dict[str, list[str]]:
    split_files = {
        "train": ["train.lst"],
        "dev": ["dev.lst", "development.lst"],
        "test": ["test.lst"],
    }
    splits: dict[str, list[str]] = {}
    for split, candidates in split_files.items():
        list_path = None
        for name in candidates:
            candidate = lists_dir / name
            if candidate.exists():
                list_path = candidate
                break
        if list_path is None:
            raise FileNotFoundError(f"Missing list file for split '{split}' under {lists_dir}.")
        uris = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        splits[split] = uris
    return splits


def load_uem_by_uri(uem_path: Path) -> dict[str, Timeline]:
    paths = sorted(uem_path.rglob("*.uem")) if uem_path.is_dir() else [uem_path]
    uems: dict[str, Timeline] = {}
    for path in paths:
        loaded = load_uem(path)
        if not isinstance(loaded, dict):
            raise TypeError("Expected load_uem to return a dictionary {uri: Timeline}.")
        uems.update(loaded)
    return uems


def load_hypothesis_json(hypothesis_json_path: Path) -> dict[str, Timeline]:
    if hypothesis_json_path.is_dir():
        paths = sorted(hypothesis_json_path.rglob("*_osd.json"))
        if not paths:
            paths = sorted(hypothesis_json_path.rglob("*.json"))
    else:
        paths = [hypothesis_json_path]

    hypothesis: dict[str, Timeline] = {}
    for path in paths:
        for record in load_json_records(path):
            uri = str(record.get("uri") or path.stem.removesuffix("_osd"))
            intervals = record.get("overlaps") or record.get("global_overlap") or []
            hypothesis[uri] = timeline_from_intervals(uri, intervals)
    return hypothesis


def safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate overlapped speech detection on mixed audio.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Dataset root containing rttm/, lists/, and uem/ subdirectories.",
    )

    parser.add_argument(
        "--hypothesis",
        type=Path,
        required=True,
        help="Predicted overlap JSON file or directory from osd.py.",
    )

    parser.add_argument("--collar", default=0.0, type=float, help="Forgiveness collar in seconds.")
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        help="Skip overlap regions during scoring. Usually keep this off for OSD.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ground_truth_root = args.ground_truth
    rttm_path = ground_truth_root / "rttm"
    lists_dir = ground_truth_root / "lists"
    uem_path = ground_truth_root / "uem"

    if not rttm_path.exists():
        raise FileNotFoundError(f"Missing RTTM directory: {rttm_path}")
    if not lists_dir.exists():
        raise FileNotFoundError(f"Missing lists directory: {lists_dir}")
    if not uem_path.exists():
        raise FileNotFoundError(f"Missing UEM directory: {uem_path}")

    rttm_by_uri = load_rttm_annotations(rttm_path)
    hypothesis_by_uri = load_hypothesis_json(args.hypothesis)
    split_uris = load_split_uris(lists_dir)
    single_hypothesis = args.hypothesis.is_file()

    uem_by_uri = load_uem_by_uri(uem_path)

    print("Evaluating overlapped speech detection")
    print(f"Lists DIR       : {lists_dir}")
    print(f"UEM DIR         : {uem_path}")
    print(f"RTTM DIR        : {rttm_path}")
    print(f"Hypothesis      : {args.hypothesis}")
    print(f"Collar          : {args.collar}")
    print()

    def evaluate_split(split: str, uris: list[str]) -> None:
        missing_ref = [uri for uri in uris if uri not in rttm_by_uri]
        if missing_ref:
            raise RuntimeError(f"Missing reference URIs in split '{split}': {missing_ref[:5]}")

        if not single_hypothesis:
            missing_hyp = [uri for uri in uris if uri not in hypothesis_by_uri]
            if missing_hyp:
                raise RuntimeError(f"Missing hypothesis URIs in split '{split}': {missing_hyp[:5]}")

        uris = [uri for uri in uris if uri in hypothesis_by_uri]
        if not uris:
            return

        precision_metric = DetectionPrecision(collar=args.collar, skip_overlap=args.skip_overlap)
        recall_metric = DetectionRecall(collar=args.collar, skip_overlap=args.skip_overlap)
        error_metric = DetectionErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)

        for uri in uris:
            reference_overlap = timeline_to_annotation(
                annotation_to_overlap_timeline(rttm_by_uri[uri]),
                label="overlap",
            )
            hypothesis_overlap = timeline_to_annotation(hypothesis_by_uri[uri], label="overlap")
            uem = uem_by_uri.get(uri)
            precision_metric(reference_overlap, hypothesis_overlap, uem=uem)
            recall_metric(reference_overlap, hypothesis_overlap, uem=uem)
            error_metric(reference_overlap, hypothesis_overlap, uem=uem)

        precision = abs(precision_metric)
        recall = abs(recall_metric)
        f1 = safe_f1(precision, recall)
        detection_error_rate = abs(error_metric)

        print(f"[{split}]")
        print(f"  Files   : {len(uris)}")
        print(f"  Precision            : {100.0 * precision:.2f}%")
        print(f"  Recall               : {100.0 * recall:.2f}%")
        print(f"  F1                   : {100.0 * f1:.2f}%")
        print(f"  Detection Error Rate : {100.0 * detection_error_rate:.2f}%")
        print()

    for split in ("train", "dev", "test"):
        uris = split_uris[split]
        if not uris:
            raise RuntimeError(f"No URIs found for split '{split}'.")
        evaluate_split(split, uris)


if __name__ == "__main__":
    main()

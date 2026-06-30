#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm, load_uem
from pyannote.metrics.detection import DetectionErrorRate, DetectionPrecision, DetectionRecall


def load_timeline(records: list[dict], uri: str) -> Timeline:
    timeline = Timeline(uri=uri)
    for item in records:
        start = float(item["start"])
        end = float(item["end"])
        if end > start:
            timeline.add(Segment(start, end))
    return timeline.support()


def rttm_overlap(rttm: Annotation) -> Timeline:
    overlap = Timeline(uri=rttm.uri)
    points = sorted({seg.start for seg, _, _ in rttm.itertracks(yield_label=True)} | {seg.end for seg, _, _ in rttm.itertracks(yield_label=True)})
    for start, end in zip(points, points[1:]):
        mid = (start + end) / 2
        active = sum(1 for seg, _, _ in rttm.itertracks(yield_label=True) if seg.start <= mid < seg.end)
        if active >= 2:
            overlap.add(Segment(start, end))
    return overlap.support()


def load_ground_truth(root: Path) -> tuple[dict[str, Annotation], dict[str, Timeline], dict[str, list[str]]]:
    rttm = {}
    for path in sorted((root / "rttm").glob("*.rttm")):
        rttm.update(load_rttm(path))

    uem = {}
    for path in sorted((root / "uem").glob("*.uem")):
        uem.update(load_uem(path))

    splits = {
        split: [line.strip() for line in (root / "lists" / f"{split}.lst").read_text(encoding="utf-8").splitlines() if line.strip()]
        for split in ("train", "dev", "test")
    }
    return rttm, uem, splits


def load_hypothesis(path: Path) -> dict[str, Timeline]:
    files = [path] if path.is_file() else sorted(path.glob("*_osd.json")) or sorted(path.glob("*.json"))
    hypothesis: dict[str, Timeline] = {}
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        records = [data] if isinstance(data, dict) else data
        for record in records:
            uri = str(record.get("uri") or file.stem.removesuffix("_osd"))
            intervals = record.get("overlaps") or record.get("global_overlap") or []
            hypothesis[uri] = load_timeline(intervals, uri)
    return hypothesis

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate overlapped speech detection.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--hypothesis", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.0)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    rttm, uem, splits = load_ground_truth(args.ground_truth)
    hypothesis = load_hypothesis(args.hypothesis)
    single_file = args.hypothesis.is_file()

    for split in ("train", "dev", "test"):
        uris = splits[split]
        if not single_file:
            uris = [uri for uri in uris if uri in hypothesis]

        precision = DetectionPrecision(collar=args.tolerance)
        recall = DetectionRecall(collar=args.tolerance)
        der = DetectionErrorRate(collar=args.tolerance)

        for uri in uris:
            ref = Annotation(uri=uri)
            for segment in rttm_overlap(rttm[uri]):
                ref[segment, "t"] = "overlap"
            hyp = Annotation(uri=uri)
            for segment in hypothesis[uri]:
                hyp[segment, "t"] = "overlap"
            precision(ref, hyp, uem=uem.get(uri))
            recall(ref, hyp, uem=uem.get(uri))
            der(ref, hyp, uem=uem.get(uri))

        p = abs(precision)
        r = abs(recall)
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        print(f"{split} ({len(uris)} files): Precision={100*p:.2f}% Recall={100*r:.2f}% F1={100*f1:.2f}% DER={100*abs(der):.2f}%")


if __name__ == "__main__":
    main()

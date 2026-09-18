#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm, load_uem
from pyannote.metrics.detection import DetectionErrorRate, DetectionPrecision, DetectionRecall
from scipy.optimize import linear_sum_assignment

MetricRow = dict[str, float]
CsvRow = dict[str, float | str]


# ==================== Necessary Function Section Start ====================

def rttm_overlap(rttm: Annotation) -> Timeline:
    overlap = Timeline(uri=rttm.uri)
    tracks = list(rttm.itertracks(yield_label=True))
    points = sorted({segment.start for segment, _, _ in tracks} | {segment.end for segment, _, _ in tracks})
    for start, end in zip(points, points[1:]):
        mid = (start + end) / 2
        if sum(1 for segment, _, _ in tracks if segment.start <= mid < segment.end) >= 2:
            overlap.add(Segment(start, end))
    return overlap.support()


def load_ground_truth(root: Path) -> tuple[dict[str, Annotation], dict[str, Timeline], dict[str, list[str]]]:
    rttm = {uri: annotation for path in sorted((root / "rttm").glob("*.rttm")) for uri, annotation in load_rttm(path).items()}
    uem = {uri: timeline for path in sorted((root / "uem").glob("*.uem")) for uri, timeline in load_uem(path).items()}
    splits = {
        split: [line.strip() for line in (root / "lists" / f"{split}.lst").read_text(encoding="utf-8").splitlines() if line.strip()]
        for split in ("train", "dev", "test")
    }
    return rttm, uem, splits


def load_hypothesis(path: Path) -> dict[str, Timeline]:
    files = [path] if path.is_file() else sorted(path.rglob("*_osd.json")) or sorted(path.rglob("*.json"))
    hypothesis_by_uri: dict[str, Timeline] = {}
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else [data]
        for record in records:
            uri = str(record.get("uri") or file.stem.removesuffix("_osd").removesuffix("_drop"))
            timeline = Timeline(uri=uri)
            for item in record.get("overlaps") or record.get("global_overlap") or record.get("overlap") or []:
                start = float(item["start"])
                end = float(item["end"])
                if end > start:
                    timeline.add(Segment(start, end))
            if uri in hypothesis_by_uri:
                hypothesis_by_uri[uri] = hypothesis_by_uri[uri].union(timeline).support()
            else:
                hypothesis_by_uri[uri] = timeline.support()
    return hypothesis_by_uri


def match_segments(
    reference_segments: list[Segment],
    hypothesis_segments: list[Segment],
    valid_fn,
    score_fn,
    method: str,
) -> list[tuple[int, int]]:
    if method == "greedy":
        candidates: list[tuple[float, int, int]] = []
        for ref_index, ref_segment in enumerate(reference_segments):
            for hyp_index, hyp_segment in enumerate(hypothesis_segments):
                if valid_fn(ref_segment, hyp_segment):
                    candidates.append((score_fn(ref_segment, hyp_segment), ref_index, hyp_index))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        matched_ref: set[int] = set()
        matched_hyp: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _, ref_index, hyp_index in candidates:
            if ref_index in matched_ref or hyp_index in matched_hyp:
                continue
            matched_ref.add(ref_index)
            matched_hyp.add(hyp_index)
            pairs.append((ref_index, hyp_index))
        return pairs

    if not reference_segments or not hypothesis_segments:
        return []

    n_ref = len(reference_segments)
    n_hyp = len(hypothesis_segments)
    cost = np.ones((n_ref + n_hyp, n_ref + n_hyp), dtype=float)
    for ref_index, ref_segment in enumerate(reference_segments):
        for hyp_index, hyp_segment in enumerate(hypothesis_segments):
            if valid_fn(ref_segment, hyp_segment):
                cost[ref_index, hyp_index] = 1.0 - float(np.clip(score_fn(ref_segment, hyp_segment), 0.0, 1.0))
    for ref_index in range(n_ref):
        cost[ref_index, n_hyp + ref_index] = 0.999999
    for hyp_index in range(n_hyp):
        cost[n_ref + hyp_index, hyp_index] = 0.999999
    cost[n_ref:, n_hyp:] = 0.0
    row_ind, col_ind = linear_sum_assignment(cost)
    return [
        (ref_index, hyp_index)
        for ref_index, hyp_index in zip(row_ind, col_ind, strict=True)
        if ref_index < n_ref and hyp_index < n_hyp and valid_fn(reference_segments[ref_index], hypothesis_segments[hyp_index])
    ]

# ==================== Necessary Function Section End ====================


# ==================== Metrics 1 ====================
def duration_based_f1(
    rttm_by_uri: dict[str, Annotation],
    hypothesis_by_uri: dict[str, Timeline],
    uem_by_uri: dict[str, Timeline],
    uris: list[str],
    *,
    duration_collar_seconds: float,
) -> MetricRow:
    error_metric = DetectionErrorRate(collar=duration_collar_seconds)
    precision_metric = DetectionPrecision(collar=duration_collar_seconds)
    recall_metric = DetectionRecall(collar=duration_collar_seconds)
    precision_detail = {"retrieved": 0.0, "relevant retrieved": 0.0}
    recall_detail = {"relevant": 0.0, "relevant retrieved": 0.0}

    def segments_for(timeline: Timeline, scoring_region: Timeline | None) -> list[Segment]:
        segments: list[Segment] = []
        for segment in timeline.support():
            if scoring_region is None:
                segments.append(segment)
                continue
            for region in scoring_region:
                start = max(segment.start, region.start)
                end = min(segment.end, region.end)
                if end > start:
                    segments.append(Segment(start, end))
        return segments

    for uri in uris:
        scoring_region = uem_by_uri.get(uri)
        reference_segments = segments_for(rttm_overlap(rttm_by_uri[uri]), scoring_region)
        hypothesis_segments = segments_for(hypothesis_by_uri.get(uri, Timeline(uri=uri)), scoring_region)
        reference = Annotation(uri=uri)
        hypothesis = Annotation(uri=uri)
        for index, segment in enumerate(reference_segments):
            reference[segment, index] = "overlap"
        for index, segment in enumerate(hypothesis_segments):
            hypothesis[segment, index] = "overlap"
        error_metric(reference, hypothesis, uem=scoring_region)
        precision_components = precision_metric.compute_components(reference, hypothesis, uem=scoring_region)
        recall_components = recall_metric.compute_components(reference, hypothesis, uem=scoring_region)
        precision_detail["retrieved"] += float(precision_components.get("retrieved", 0.0))
        precision_detail["relevant retrieved"] += float(precision_components.get("relevant retrieved", 0.0))
        recall_detail["relevant"] += float(recall_components.get("relevant", 0.0))
        recall_detail["relevant retrieved"] += float(recall_components.get("relevant retrieved", 0.0))

    components = error_metric[:]
    relevant = float(components.get("total", 0.0))
    fp = float(components.get("false alarm", 0.0))
    fn = float(components.get("missed detection", components.get("miss", 0.0)))
    precision = float(precision_metric.compute_metric(precision_detail))
    recall = float(recall_metric.compute_metric(recall_detail))
    return {
        "error_rate": float(abs(error_metric)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": 2 * float(precision) * float(recall) / (float(precision) + float(recall)) if float(precision) + float(recall) > 0 else 0.0,
        "miss_rate": fn / relevant if relevant > 0 else 0.0,
        "false_alarm_rate": fp / relevant if relevant > 0 else 0.0,
        "fp_seconds": fp,
        "fn_seconds": fn,
        "reference_seconds": relevant,
    }

# ==================== Metrics 2 ====================
def event_based_f1(
    rttm_by_uri: dict[str, Annotation],
    hypothesis_by_uri: dict[str, Timeline],
    uem_by_uri: dict[str, Timeline],
    uris: list[str],
    *,
    event_onset_collar_seconds: float,
    event_offset_collar_seconds: float,
    event_offset_collar_rate: float,
    event_matching_rule: str,
    event_matching_method: str,
    event_hypothesis_min_duration_seconds: float,
    event_hypothesis_merge_gap_seconds: float,
) -> MetricRow:
    event_totals = {key: 0.0 for key in ("tp_count", "fp_count", "fn_count", "reference_count", "hypothesis_count")}

    def segments_for(timeline: Timeline, scoring_region: Timeline | None) -> list[Segment]:
        segments: list[Segment] = []
        for segment in timeline.support():
            if scoring_region is None:
                segments.append(segment)
                continue
            for region in scoring_region:
                start = max(segment.start, region.start)
                end = min(segment.end, region.end)
                if end > start:
                    segments.append(Segment(start, end))
        segments.sort(key=lambda segment: (segment.start, segment.end))
        if not segments:
            return []
        merged = [segments[0]]
        for segment in segments[1:]:
            current = merged[-1]
            if segment.start - current.end <= event_hypothesis_merge_gap_seconds:
                merged[-1] = Segment(current.start, max(current.end, segment.end))
            else:
                merged.append(segment)
        return [segment for segment in merged if segment.duration >= event_hypothesis_min_duration_seconds]

    def valid(reference: Segment, hypothesis: Segment) -> bool:
        onset_error = abs(hypothesis.start - reference.start)
        if onset_error > event_onset_collar_seconds:
            return False
        if event_matching_rule == "onset":
            return True
        return abs(hypothesis.end - reference.end) <= max(event_offset_collar_seconds, event_offset_collar_rate * reference.duration)

    def score(reference: Segment, hypothesis: Segment) -> float:
        return 1.0 / (1.0 + abs(hypothesis.start - reference.start) + abs(hypothesis.end - reference.end))

    for uri in uris:
        scoring_region = uem_by_uri.get(uri)
        reference_segments = segments_for(rttm_overlap(rttm_by_uri[uri]), scoring_region)
        hypothesis_segments = segments_for(hypothesis_by_uri.get(uri, Timeline(uri=uri)), scoring_region)
        pairs = match_segments(reference_segments, hypothesis_segments, valid, score, event_matching_method)
        matched = len(pairs)
        reference_count = len(reference_segments)
        hypothesis_count = len(hypothesis_segments)
        event_totals["tp_count"] += float(matched)
        event_totals["fp_count"] += float(hypothesis_count - matched)
        event_totals["fn_count"] += float(reference_count - matched)
        event_totals["reference_count"] += float(reference_count)
        event_totals["hypothesis_count"] += float(hypothesis_count)

    precision = event_totals["tp_count"] / event_totals["hypothesis_count"] if event_totals["hypothesis_count"] > 0 else 0.0
    recall = event_totals["tp_count"] / event_totals["reference_count"] if event_totals["reference_count"] > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        **event_totals,
    }

# ==================== Metrics 3 ====================
def pairwise_intersection_based_f1(
    rttm_by_uri: dict[str, Annotation],
    hypothesis_by_uri: dict[str, Timeline],
    uem_by_uri: dict[str, Timeline],
    uris: list[str],
    *,
    intersection_dtc_threshold: float,
    intersection_gtc_threshold: float,
    intersection_matching_method: str,
    intersection_hypothesis_min_duration_seconds: float,
    intersection_hypothesis_merge_gap_seconds: float,
) -> MetricRow:
    intersection_totals = {
        key: 0.0
        for key in ("tp_count", "fp_count", "fn_count", "reference_count", "hypothesis_count", "intersection_seconds", "overlap_seconds", "reference_seconds", "hypothesis_seconds")
    }

    def segments_for(timeline: Timeline, scoring_region: Timeline | None) -> list[Segment]:
        segments: list[Segment] = []
        for segment in timeline.support():
            if scoring_region is None:
                segments.append(segment)
                continue
            for region in scoring_region:
                start = max(segment.start, region.start)
                end = min(segment.end, region.end)
                if end > start:
                    segments.append(Segment(start, end))
        segments.sort(key=lambda segment: (segment.start, segment.end))
        if not segments:
            return []
        merged = [segments[0]]
        for segment in segments[1:]:
            current = merged[-1]
            if segment.start - current.end <= intersection_hypothesis_merge_gap_seconds:
                merged[-1] = Segment(current.start, max(current.end, segment.end))
            else:
                merged.append(segment)
        return [segment for segment in merged if segment.duration >= intersection_hypothesis_min_duration_seconds]

    def intersect(reference: Segment, hypothesis: Segment) -> float:
        return max(0.0, min(reference.end, hypothesis.end) - max(reference.start, hypothesis.start))

    def valid(reference: Segment, hypothesis: Segment) -> bool:
        inter = intersect(reference, hypothesis)
        return inter > 0.0 and inter / hypothesis.duration >= intersection_dtc_threshold and inter / reference.duration >= intersection_gtc_threshold

    def score(reference: Segment, hypothesis: Segment) -> float:
        inter = intersect(reference, hypothesis)
        return 0.5 * (inter / reference.duration + inter / hypothesis.duration)

    def overlap_duration(reference_segments: list[Segment], hypothesis_segments: list[Segment]) -> float:
        total = 0.0
        i = j = 0
        while i < len(reference_segments) and j < len(hypothesis_segments):
            start = max(reference_segments[i].start, hypothesis_segments[j].start)
            end = min(reference_segments[i].end, hypothesis_segments[j].end)
            if end > start:
                total += end - start
            if reference_segments[i].end <= hypothesis_segments[j].end:
                i += 1
            else:
                j += 1
        return total

    for uri in uris:
        scoring_region = uem_by_uri.get(uri)
        reference_segments = segments_for(rttm_overlap(rttm_by_uri[uri]), scoring_region)
        hypothesis_segments = segments_for(hypothesis_by_uri.get(uri, Timeline(uri=uri)), scoring_region)
        pairs = match_segments(reference_segments, hypothesis_segments, valid, score, intersection_matching_method)
        matched = len(pairs)
        reference_count = len(reference_segments)
        hypothesis_count = len(hypothesis_segments)
        reference_seconds = float(sum(segment.duration for segment in reference_segments))
        hypothesis_seconds = float(sum(segment.duration for segment in hypothesis_segments))
        overlap_seconds = overlap_duration(reference_segments, hypothesis_segments)
        intersection_seconds = float(sum(intersect(reference_segments[ref_index], hypothesis_segments[hyp_index]) for ref_index, hyp_index in pairs))
        intersection_totals["tp_count"] += float(matched)
        intersection_totals["fp_count"] += float(hypothesis_count - matched)
        intersection_totals["fn_count"] += float(reference_count - matched)
        intersection_totals["reference_count"] += float(reference_count)
        intersection_totals["hypothesis_count"] += float(hypothesis_count)
        intersection_totals["intersection_seconds"] += intersection_seconds
        intersection_totals["overlap_seconds"] += overlap_seconds
        intersection_totals["reference_seconds"] += reference_seconds
        intersection_totals["hypothesis_seconds"] += hypothesis_seconds

    precision = intersection_totals["tp_count"] / intersection_totals["hypothesis_count"] if intersection_totals["hypothesis_count"] > 0 else 0.0
    recall = intersection_totals["tp_count"] / intersection_totals["reference_count"] if intersection_totals["reference_count"] > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        **intersection_totals,
        "coverage": intersection_totals["overlap_seconds"] / intersection_totals["reference_seconds"] if intersection_totals["reference_seconds"] > 0 else 0.0,
        "purity": intersection_totals["overlap_seconds"] / intersection_totals["hypothesis_seconds"] if intersection_totals["hypothesis_seconds"] > 0 else 0.0,
    }


# ==================== Metrics 4 ====================
def boundary_error(
    rttm_by_uri: dict[str, Annotation],
    hypothesis_by_uri: dict[str, Timeline],
    uem_by_uri: dict[str, Timeline],
    uris: list[str],
    *,
    boundary_rule: str,
    boundary_matching_method: str,
    boundary_onset_collar_seconds: float,
    boundary_offset_collar_seconds: float,
    boundary_offset_collar_rate: float,
    boundary_min_iou: float,
    boundary_min_intersection_seconds: float,
    boundary_max_onset_distance_seconds: float,
    boundary_hypothesis_min_duration_seconds: float,
    boundary_hypothesis_merge_gap_seconds: float,
    boundary_percentiles: list[int],
) -> MetricRow:
    boundary_totals = {key: 0.0 for key in ("tp_count", "fp_count", "fn_count", "reference_count", "hypothesis_count")}
    onset_errors: list[np.ndarray] = []
    offset_errors: list[np.ndarray] = []
    duration_errors: list[np.ndarray] = []

    def segments_for(timeline: Timeline, scoring_region: Timeline | None) -> list[Segment]:
        segments: list[Segment] = []
        for segment in timeline.support():
            if scoring_region is None:
                segments.append(segment)
                continue
            for region in scoring_region:
                start = max(segment.start, region.start)
                end = min(segment.end, region.end)
                if end > start:
                    segments.append(Segment(start, end))
        segments.sort(key=lambda segment: (segment.start, segment.end))
        if not segments:
            return []
        merged = [segments[0]]
        for segment in segments[1:]:
            current = merged[-1]
            if segment.start - current.end <= boundary_hypothesis_merge_gap_seconds:
                merged[-1] = Segment(current.start, max(current.end, segment.end))
            else:
                merged.append(segment)
        return [segment for segment in merged if segment.duration >= boundary_hypothesis_min_duration_seconds]

    def intersect(reference: Segment, hypothesis: Segment) -> float:
        return max(0.0, min(reference.end, hypothesis.end) - max(reference.start, hypothesis.start))

    def valid(reference: Segment, hypothesis: Segment) -> bool:
        inter = intersect(reference, hypothesis)
        if boundary_rule == "collar":
            return abs(hypothesis.start - reference.start) <= boundary_onset_collar_seconds and abs(hypothesis.end - reference.end) <= max(
                boundary_offset_collar_seconds,
                boundary_offset_collar_rate * reference.duration,
            )
        if boundary_rule == "iou":
            union = reference.duration + hypothesis.duration - inter
            return inter / union >= boundary_min_iou if union > 0 else False
        if boundary_rule == "intersection":
            return inter >= boundary_min_intersection_seconds
        if boundary_rule == "nearest_onset":
            return abs(hypothesis.start - reference.start) <= boundary_max_onset_distance_seconds
        return False

    def score(reference: Segment, hypothesis: Segment) -> float:
        inter = intersect(reference, hypothesis)
        if boundary_rule == "iou":
            union = reference.duration + hypothesis.duration - inter
            return inter / union if union > 0 else 0.0
        if boundary_rule == "intersection":
            return inter / max(reference.duration, hypothesis.duration)
        return 1.0 / (1.0 + abs(hypothesis.start - reference.start))

    def concat(values: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(values) if values else np.asarray([], dtype=float)

    def abs_stats(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            stats = {"mean": float("nan"), "mean_abs": float("nan"), "median_abs": float("nan")}
            for percentile in boundary_percentiles:
                stats[f"p{int(percentile)}_abs"] = float("nan")
            return stats
        absolute = np.abs(values)
        stats = {
            "mean": float(np.mean(values)),
            "mean_abs": float(np.mean(absolute)),
            "median_abs": float(np.median(absolute)),
        }
        for percentile in boundary_percentiles:
            stats[f"p{int(percentile)}_abs"] = float(np.percentile(absolute, percentile))
        return stats

    for uri in uris:
        scoring_region = uem_by_uri.get(uri)
        reference_segments = segments_for(rttm_overlap(rttm_by_uri[uri]), scoring_region)
        hypothesis_segments = segments_for(hypothesis_by_uri.get(uri, Timeline(uri=uri)), scoring_region)
        pairs = match_segments(reference_segments, hypothesis_segments, valid, score, boundary_matching_method)
        matched = len(pairs)
        reference_count = len(reference_segments)
        hypothesis_count = len(hypothesis_segments)
        boundary_totals["tp_count"] += float(matched)
        boundary_totals["fp_count"] += float(hypothesis_count - matched)
        boundary_totals["fn_count"] += float(reference_count - matched)
        boundary_totals["reference_count"] += float(reference_count)
        boundary_totals["hypothesis_count"] += float(hypothesis_count)
        onset_errors.append(np.asarray([hypothesis_segments[h].start - reference_segments[r].start for r, h in pairs], dtype=float))
        offset_errors.append(np.asarray([hypothesis_segments[h].end - reference_segments[r].end for r, h in pairs], dtype=float))
        duration_errors.append(np.asarray([hypothesis_segments[h].duration - reference_segments[r].duration for r, h in pairs], dtype=float))

    onset_stats = abs_stats(concat(onset_errors))
    offset_stats = abs_stats(concat(offset_errors))
    duration_stats = abs_stats(concat(duration_errors))
    precision = boundary_totals["tp_count"] / boundary_totals["hypothesis_count"] if boundary_totals["hypothesis_count"] > 0 else 0.0
    recall = boundary_totals["tp_count"] / boundary_totals["reference_count"] if boundary_totals["reference_count"] > 0 else 0.0
    boundary = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        "match_rate": recall,
        **boundary_totals,
        "onset_mean_error_seconds": onset_stats["mean"],
        "onset_mean_abs_error_seconds": onset_stats["mean_abs"],
        "onset_median_abs_error_seconds": onset_stats["median_abs"],
        "offset_mean_error_seconds": offset_stats["mean"],
        "offset_mean_abs_error_seconds": offset_stats["mean_abs"],
        "offset_median_abs_error_seconds": offset_stats["median_abs"],
        "duration_mean_error_seconds": duration_stats["mean"],
        "duration_mean_abs_error_seconds": duration_stats["mean_abs"],
        "duration_median_abs_error_seconds": duration_stats["median_abs"],
    }
    for percentile in boundary_percentiles:
        suffix = f"p{int(percentile)}"
        boundary[f"onset_{suffix}_abs_error_seconds"] = onset_stats[f"{suffix}_abs"]
        boundary[f"offset_{suffix}_abs_error_seconds"] = offset_stats[f"{suffix}_abs"]
        boundary[f"duration_{suffix}_abs_error_seconds"] = duration_stats[f"{suffix}_abs"]
    return boundary


def render_summary_table(rows: list[CsvRow]) -> None:
    def fmt(value: object) -> str:
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    def metric_block(title: str, entries: list[tuple[str, object]], label_width: int) -> list[str]:
        lines = [title]
        lines.extend(f"  {label:<{label_width}} {fmt(value)}" for label, value in entries)
        return lines

    for row in rows:
        entry_groups = [
            [
                ("precision", row["duration_based_f1_precision"]),
                ("recall", row["duration_based_f1_recall"]),
                ("f1", row["duration_based_f1_f1"]),
                ("error_rate", row["duration_based_f1_error_rate"]),
                ("miss_rate", row["duration_based_f1_miss_rate"]),
                ("false_alarm_rate", row["duration_based_f1_false_alarm_rate"]),
            ],
            [
                ("precision", row["event_based_f1_precision"]),
                ("recall", row["event_based_f1_recall"]),
                ("f1", row["event_based_f1_f1"]),
            ],
            [
                ("precision", row["pairwise_intersection_based_f1_precision"]),
                ("recall", row["pairwise_intersection_based_f1_recall"]),
                ("f1", row["pairwise_intersection_based_f1_f1"]),
                ("coverage", row["pairwise_intersection_based_f1_coverage"]),
                ("purity", row["pairwise_intersection_based_f1_purity"]),
            ],
            [
                ("precision", row["boundary_error_precision"]),
                ("recall", row["boundary_error_recall"]),
                ("f1", row["boundary_error_f1"]),
                ("match_rate", row["boundary_error_match_rate"]),
                ("tp_count", row["boundary_error_tp_count"]),
                ("fp_count", row["boundary_error_fp_count"]),
                ("fn_count", row["boundary_error_fn_count"]),
                ("onset_mean_error_seconds", row["boundary_error_onset_mean_error_seconds"]),
                ("onset_mean_abs_error_seconds", row["boundary_error_onset_mean_abs_error_seconds"]),
                ("onset_median_abs_error_seconds", row["boundary_error_onset_median_abs_error_seconds"]),
                ("onset_p90_abs_error_seconds", row["boundary_error_onset_p90_abs_error_seconds"]),
                ("offset_mean_error_seconds", row["boundary_error_offset_mean_error_seconds"]),
                ("offset_mean_abs_error_seconds", row["boundary_error_offset_mean_abs_error_seconds"]),
                ("offset_median_abs_error_seconds", row["boundary_error_offset_median_abs_error_seconds"]),
                ("offset_p90_abs_error_seconds", row["boundary_error_offset_p90_abs_error_seconds"]),
                ("duration_mean_error_seconds", row["boundary_error_duration_mean_error_seconds"]),
                ("duration_mean_abs_error_seconds", row["boundary_error_duration_mean_abs_error_seconds"]),
                ("duration_median_abs_error_seconds", row["boundary_error_duration_median_abs_error_seconds"]),
                ("duration_p90_abs_error_seconds", row["boundary_error_duration_p90_abs_error_seconds"]),
            ],
        ]
        label_width = max(len(label) for group in entry_groups for label, _ in group)
        lines: list[str] = [f"Split: {row['split']}"]
        lines.extend(metric_block("Duration-based F1", entry_groups[0], label_width))
        lines.extend(metric_block("Event-based F1", entry_groups[1], label_width))
        lines.extend(metric_block("Pairwise Intersection-based F1", entry_groups[2], label_width))
        lines.extend(metric_block("Boundary Error", entry_groups[3], label_width))
        width = max(len(line) for line in lines)
        print()
        print("┌" + "─" * (width + 2) + "┐")
        for line in lines:
            print(f"│ {line.ljust(width)} │")
        print("└" + "─" * (width + 2) + "┘")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate overlap speech detection with duration, event, intersection, and boundary metrics.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Root directory containing ground-truth lists, RTTM, and UEM files.")
    parser.add_argument("--hypothesis", type=Path, required=True, help="Directory or file containing prediction JSON files.")
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV file path to write the summary.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output CSV if it already exists.")

    parser.add_argument("--duration-collar-seconds", type=float, default=0.0, help="Duration-based collar in seconds.")

    parser.add_argument("--event-onset-collar-seconds", type=float, default=0.25, help="Event onset collar in seconds.")
    parser.add_argument("--event-offset-collar-seconds", type=float, default=0.25, help="Event offset collar floor in seconds.")
    parser.add_argument("--event-offset-collar-rate", type=float, default=0.2, help="Event offset collar ratio of reference duration.")
    parser.add_argument("--event-matching-rule", choices=("onset", "onset_offset"), default="onset_offset", help="Event matching rule.")
    parser.add_argument("--event-matching-method", choices=("greedy", "hungarian"), default="hungarian", help="Event matching method.")
    parser.add_argument("--event-hypothesis-min-duration-seconds", type=float, default=0.0, help="Minimum hypothesis event duration for event metrics.")
    parser.add_argument("--event-hypothesis-merge-gap-seconds", type=float, default=0.0, help="Maximum hypothesis gap to merge for event metrics.")

    parser.add_argument("--intersection-dtc-threshold", type=float, default=0.5, help="Intersection DTC threshold.")
    parser.add_argument("--intersection-gtc-threshold", type=float, default=0.5, help="Intersection GTC threshold.")
    parser.add_argument("--intersection-matching-method", choices=("greedy", "hungarian"), default="hungarian", help="Intersection matching method.")
    parser.add_argument("--intersection-hypothesis-min-duration-seconds", type=float, default=0.0, help="Minimum hypothesis event duration for intersection metrics.")
    parser.add_argument("--intersection-hypothesis-merge-gap-seconds", type=float, default=0.0, help="Maximum hypothesis gap to merge for intersection metrics.")

    parser.add_argument("--boundary-matching-rule", choices=("collar", "iou", "intersection", "nearest_onset"), default="intersection", help="Boundary matching rule.")
    parser.add_argument("--boundary-matching-method", choices=("greedy", "hungarian"), default="hungarian", help="Boundary matching method.")
    parser.add_argument("--boundary-onset-collar-seconds", type=float, default=0.25, help="Boundary onset collar in seconds.")
    parser.add_argument("--boundary-offset-collar-seconds", type=float, default=0.25, help="Boundary offset collar floor in seconds.")
    parser.add_argument("--boundary-offset-collar-rate", type=float, default=0.2, help="Boundary offset collar ratio of reference duration.")
    parser.add_argument("--boundary-min-iou", type=float, default=0.1, help="Boundary IoU threshold.")
    parser.add_argument("--boundary-min-intersection-seconds", type=float, default=0.05, help="Boundary intersection threshold in seconds.")
    parser.add_argument("--boundary-max-onset-distance-seconds", type=float, default=0.5, help="Boundary onset distance limit for nearest-onset matching.")
    parser.add_argument("--boundary-hypothesis-min-duration-seconds", type=float, default=0.0, help="Minimum hypothesis event duration for boundary metrics.")
    parser.add_argument("--boundary-hypothesis-merge-gap-seconds", type=float, default=0.0, help="Maximum hypothesis gap to merge for boundary metrics.")
    parser.add_argument("--boundary-percentiles", type=int, nargs="+", default=(50, 90), help="Integer percentiles for boundary absolute-error summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv
    if output_csv.exists() and not args.force:
        raise FileExistsError(f"{output_csv} already exists. Use --force to overwrite it.")

    print("🚀 OSD evaluation")
    print(f"   • Ground truth: {args.ground_truth}")
    print(f"   • Hypothesis: {args.hypothesis}")

    rttm_by_uri, uem_by_uri, splits = load_ground_truth(args.ground_truth)
    hypothesis_by_uri = load_hypothesis(args.hypothesis)
    settings = {
        "duration_collar_seconds": args.duration_collar_seconds,
        "event_matching_rule": args.event_matching_rule,
        "event_onset_collar_seconds": args.event_onset_collar_seconds,
        "event_offset_collar_seconds": args.event_offset_collar_seconds,
        "event_offset_collar_rate": args.event_offset_collar_rate,
        "event_matching_method": args.event_matching_method,
        "event_hypothesis_min_duration_seconds": args.event_hypothesis_min_duration_seconds,
        "event_hypothesis_merge_gap_seconds": args.event_hypothesis_merge_gap_seconds,
        "intersection_dtc_threshold": args.intersection_dtc_threshold,
        "intersection_gtc_threshold": args.intersection_gtc_threshold,
        "intersection_matching_method": args.intersection_matching_method,
        "intersection_hypothesis_min_duration_seconds": args.intersection_hypothesis_min_duration_seconds,
        "intersection_hypothesis_merge_gap_seconds": args.intersection_hypothesis_merge_gap_seconds,
        "boundary_rule": args.boundary_matching_rule,
        "boundary_matching_method": args.boundary_matching_method,
        "boundary_onset_collar_seconds": args.boundary_onset_collar_seconds,
        "boundary_offset_collar_seconds": args.boundary_offset_collar_seconds,
        "boundary_offset_collar_rate": args.boundary_offset_collar_rate,
        "boundary_min_iou": args.boundary_min_iou,
        "boundary_min_intersection_seconds": args.boundary_min_intersection_seconds,
        "boundary_max_onset_distance_seconds": args.boundary_max_onset_distance_seconds,
        "boundary_hypothesis_min_duration_seconds": args.boundary_hypothesis_min_duration_seconds,
        "boundary_hypothesis_merge_gap_seconds": args.boundary_hypothesis_merge_gap_seconds,
        "boundary_percentiles": ",".join(str(value) for value in args.boundary_percentiles),
    }

    rows: list[CsvRow] = []
    for split in ("train", "dev", "test"):
        uris = [uri for uri in splits.get(split, []) if uri in rttm_by_uri]
        split_metrics = {
            "duration_based_f1": duration_based_f1(
                rttm_by_uri,
                hypothesis_by_uri,
                uem_by_uri,
                uris,
                duration_collar_seconds=args.duration_collar_seconds,
            ),
            "event_based_f1": event_based_f1(
                rttm_by_uri,
                hypothesis_by_uri,
                uem_by_uri,
                uris,
                event_onset_collar_seconds=args.event_onset_collar_seconds,
                event_offset_collar_seconds=args.event_offset_collar_seconds,
                event_offset_collar_rate=args.event_offset_collar_rate,
                event_matching_rule=args.event_matching_rule,
                event_matching_method=args.event_matching_method,
                event_hypothesis_min_duration_seconds=args.event_hypothesis_min_duration_seconds,
                event_hypothesis_merge_gap_seconds=args.event_hypothesis_merge_gap_seconds,
            ),
            "pairwise_intersection_based_f1": pairwise_intersection_based_f1(
                rttm_by_uri,
                hypothesis_by_uri,
                uem_by_uri,
                uris,
                intersection_dtc_threshold=args.intersection_dtc_threshold,
                intersection_gtc_threshold=args.intersection_gtc_threshold,
                intersection_matching_method=args.intersection_matching_method,
                intersection_hypothesis_min_duration_seconds=args.intersection_hypothesis_min_duration_seconds,
                intersection_hypothesis_merge_gap_seconds=args.intersection_hypothesis_merge_gap_seconds,
            ),
            "boundary_error": boundary_error(
                rttm_by_uri,
                hypothesis_by_uri,
                uem_by_uri,
                uris,
                boundary_rule=args.boundary_matching_rule,
                boundary_matching_method=args.boundary_matching_method,
                boundary_onset_collar_seconds=args.boundary_onset_collar_seconds,
                boundary_offset_collar_seconds=args.boundary_offset_collar_seconds,
                boundary_offset_collar_rate=args.boundary_offset_collar_rate,
                boundary_min_iou=args.boundary_min_iou,
                boundary_min_intersection_seconds=args.boundary_min_intersection_seconds,
                boundary_max_onset_distance_seconds=args.boundary_max_onset_distance_seconds,
                boundary_hypothesis_min_duration_seconds=args.boundary_hypothesis_min_duration_seconds,
                boundary_hypothesis_merge_gap_seconds=args.boundary_hypothesis_merge_gap_seconds,
                boundary_percentiles=list(args.boundary_percentiles),
            ),
        }

        row: CsvRow = {"split": split, **settings}
        for metric_name, values in split_metrics.items():
            for key, value in values.items():
                row[f"{metric_name}_{key}"] = float(value)
        rows.append(row)

    columns = ["split"]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    render_summary_table(rows)
    print()
    print("✅ OSD evaluation completed.")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()

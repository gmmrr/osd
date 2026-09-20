#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from utils.visualization import (
    AudioVisualizationRecord,
    SpeakerCountInterval,
    audio_duration,
    build_speaker_count_intervals,
    display_relative,
    find_wav_paths,
    load_json_object,
    load_rttm_directory,
    make_panel,
    make_record,
    sort_records,
    sort_speakers,
    run_visualization,
)


@dataclass(frozen=True)
class OSDCAnnotation:
    json_path: Path
    classes: list[str]
    intervals: list[SpeakerCountInterval]


def load_osdc_annotations(path: Path) -> dict[str, OSDCAnnotation]:
    json_paths = sorted(path.glob("*_osdc.json")) if path.is_dir() else [path]
    annotations: dict[str, OSDCAnnotation] = {}

    for json_path in json_paths:
        data = load_json_object(json_path)
        uri = str(data.get("uri") or json_path.stem.removesuffix("_osdc"))
        intervals = [
            SpeakerCountInterval(
                start=float(segment["start"]),
                end=float(segment["end"]),
                count=int(segment["count"]),
            )
            for segment in data.get("counts", [])
            if float(segment["end"]) > float(segment["start"])
        ]
        annotations[uri] = OSDCAnnotation(
            json_path=json_path.resolve(),
            classes=[str(label) for label in data.get("classes", [])],
            intervals=sorted(intervals, key=lambda interval: (interval.start, interval.end)),
        )

    return annotations


def cover_audio_duration(
    intervals: list[SpeakerCountInterval],
    duration: float,
) -> list[SpeakerCountInterval]:
    covered: list[SpeakerCountInterval] = []
    cursor = 0.0

    for interval in intervals:
        start = max(cursor, min(duration, interval.start))
        end = min(duration, interval.end)
        if start > cursor:
            covered.append(SpeakerCountInterval(cursor, start, 0))
        if end > start:
            covered.append(SpeakerCountInterval(start, end, interval.count))
            cursor = end

    if cursor < duration:
        covered.append(SpeakerCountInterval(cursor, duration, 0))

    merged: list[SpeakerCountInterval] = []
    for interval in covered:
        if merged and merged[-1].count == interval.count:
            previous = merged[-1]
            merged[-1] = SpeakerCountInterval(previous.start, interval.end, interval.count)
        else:
            merged.append(interval)
    return merged


def build_records(
    audio_root: Path,
    osdc_json_root: Path,
    rttm_dir: Path | None = None,
) -> list[AudioVisualizationRecord]:
    osdc_by_uri = load_osdc_annotations(osdc_json_root)
    ground_truth_by_uri = load_rttm_directory(rttm_dir) if rttm_dir is not None else None
    records: list[AudioVisualizationRecord] = []

    for audio_path in find_wav_paths(audio_root, prefer_audio_dir=True):
        annotation = osdc_by_uri.get(audio_path.stem)
        if annotation is None:
            continue

        duration = audio_duration(audio_path)
        count_intervals = cover_audio_duration(annotation.intervals, duration)
        max_observed_count = max((interval.count for interval in count_intervals), default=0)
        max_supported_count = max(len(annotation.classes) - 1, max_observed_count)
        scale_label = annotation.classes[-1] if annotation.classes else str(max_supported_count)

        panels = []
        if ground_truth_by_uri is not None:
            speaker_segments = ground_truth_by_uri.get(audio_path.stem, [])
            speakers = sort_speakers(segment.speaker for segment in speaker_segments)
            ground_truth_counts, max_ground_truth_count = build_speaker_count_intervals(
                speaker_segments,
                duration,
            )
            clipped_ground_truth_counts = [
                SpeakerCountInterval(interval.start, interval.end, min(interval.count, max_supported_count))
                for interval in ground_truth_counts
            ]
            panels.append(
                make_panel(
                    tags=[
                        "Ground Truth",
                        f"speakers={len(speakers)}",
                        f"max count={max_ground_truth_count}",
                        f"scale=0–{scale_label}",
                    ],
                    note=display_relative(rttm_dir),
                    colored_speakers=speaker_segments,
                    use_subtle_background=True,
                    speakers=speakers,
                    speaker_count_intervals=clipped_ground_truth_counts,
                    max_speaker_count=max_supported_count,
                    always_show_speaker_count_strip=True,
                )
            )

        panels.append(
            make_panel(
                tags=[
                    "OSDC Prediction",
                    f"segments={len(annotation.intervals)}",
                    f"max count={max_observed_count}",
                    f"scale=0–{scale_label}",
                ],
                note=display_relative(annotation.json_path),
                use_subtle_background=True,
                speaker_count_intervals=count_intervals,
                max_speaker_count=max_supported_count,
                always_show_speaker_count_strip=True,
            )
        )
        records.append(make_record(audio_path, panels, display_relative(audio_path)))

    if not records:
        raise RuntimeError("No shared URIs between audio files and OSDC JSON.")
    return sort_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize OSDC predictions for the same audio.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing the original audio files.")
    parser.add_argument("--osdc-json", type=Path, required=True, help="OSDC JSON file or directory containing *_osdc.json.")
    parser.add_argument("--rttm-dir", type=Path, help="RTTM directory for ground-truth comparison mode.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(args.audio_dir, args.osdc_json, args.rttm_dir)
    title = "OSDC Demo Visualization" if args.rttm_dir is not None else "OSDC Visualization"
    run_visualization(title, records)


if __name__ == "__main__":
    main()

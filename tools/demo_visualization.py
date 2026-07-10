#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from utils.visualization import (
    AudioVisualizationRecord,
    display_relative,
    find_wav_paths,
    format_optional_float_tag,
    make_panel,
    make_record,
    load_osd_annotations,
    load_rttm_segments,
    sort_records,
    run_visualization,
    sort_speakers,
)


def build_records(ground_truth_root: Path, osd_json_root: Path) -> list[AudioVisualizationRecord]:
    segments_by_uri = load_rttm_segments(ground_truth_root)
    osd_by_uri = load_osd_annotations(osd_json_root)
    records: list[AudioVisualizationRecord] = []

    for audio_path in find_wav_paths(ground_truth_root, prefer_audio_dir=True):
        uri = audio_path.stem
        osd_annotation = osd_by_uri.get(uri)
        if osd_annotation is None:
            continue

        speaker_segments = segments_by_uri.get(uri, [])
        speakers = sort_speakers(segment.speaker for segment in speaker_segments)
        panels = [
            make_panel(
                tags=["Ground Truth", f"speakers={len(speakers)}"],
                note=display_relative(ground_truth_root / "rttm"),
                colored_speakers=speaker_segments,
                use_subtle_background=True,
                speakers=speakers,
            ),
            make_panel(
                tags=[
                    "OSD Prediction",
                    format_optional_float_tag("onset", osd_annotation.onset),
                    format_optional_float_tag("offset", osd_annotation.offset),
                ],
                note=display_relative(osd_annotation.json_path),
                overlay_intervals=osd_annotation.intervals,
                use_subtle_background=True,
            ),
        ]
        records.append(make_record(audio_path, panels, display_relative(audio_path)))

    if not records:
        raise RuntimeError("No shared URIs between ground-truth audio and OSD JSON.")
    return sort_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize ground-truth and OSD predictions for the same audio.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Dataset root containing audio/ and rttm/.")
    parser.add_argument("--osd-json", type=Path, required=True, help="OSD JSON file or directory containing *_osd.json.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(args.ground_truth, args.osd_json)
    run_visualization("Demo Visualization", records)


if __name__ == "__main__":
    main()

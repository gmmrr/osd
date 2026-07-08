#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from utils.visualization import (
    AudioVisualizationRecord,
    display_relative,
    find_wav_paths,
    format_optional_float_tag,
    load_osd_annotations,
    make_panel,
    make_record,
    sort_records,
    run_visualization,
)


def build_records(audio_root: Path, osd_json_root: Path) -> list[AudioVisualizationRecord]:
    osd_by_uri = load_osd_annotations(osd_json_root)
    records: list[AudioVisualizationRecord] = []

    for audio_path in find_wav_paths(audio_root, prefer_audio_dir=True):
        osd_annotation = osd_by_uri.get(audio_path.stem)
        if osd_annotation is None:
            continue

        panel = make_panel(
            tags=[
                "OSD Prediction",
                format_optional_float_tag("onset", osd_annotation.onset),
                format_optional_float_tag("offset", osd_annotation.offset),
                f"segments={len(osd_annotation.intervals)}",
            ],
            note=display_relative(osd_annotation.json_path),
            overlay_intervals=osd_annotation.intervals,
            use_subtle_background=True,
        )
        records.append(make_record(audio_path, [panel], display_relative(audio_path)))

    if not records:
        raise RuntimeError("No shared URIs between audio files and OSD JSON.")
    return sort_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize OSD predictions for the same audio.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing the original audio files.")
    parser.add_argument("--osd-json", type=Path, required=True, help="OSD JSON file or directory containing *_osd.json.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(args.audio_dir, args.osd_json)
    run_visualization("OSD Visualization", records)


if __name__ == "__main__":
    main()

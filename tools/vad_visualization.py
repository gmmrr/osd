#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from utils.visualization import (
    AudioVisualizationRecord,
    TimeInterval,
    display_relative,
    format_optional_float_tag,
    load_json_object,
    make_panel,
    make_record,
    require_directory,
    sort_records,
    run_visualization,
)


def build_records(json_root: Path) -> list[AudioVisualizationRecord]:
    root = require_directory(json_root)
    annotations_by_audio: dict[Path, list[tuple[Path, str | None, float | None, list[TimeInterval]]]] = {}

    for json_path in sorted(path.resolve() for path in root.rglob("*_std_nml_vad.json") if path.is_file()):
        data = load_json_object(json_path)
        parameters = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
        model_value = data.get("model") if data.get("model") is not None else parameters.get("model")
        threshold_value = data.get("threshold") if data.get("threshold") is not None else parameters.get("threshold")
        model_name = str(model_value) if model_value is not None else None
        threshold = float(threshold_value) if threshold_value is not None else None
        intervals = [
            TimeInterval(start=float(segment["start"]), end=float(segment["end"]))
            for segment in data.get("segments", [])
            if float(segment["end"]) > float(segment["start"])
        ]

        audio_path = json_path.with_name(json_path.name.removesuffix("_vad.json") + ".wav").resolve()
        annotations_by_audio.setdefault(audio_path, []).append(
            (json_path, model_name, threshold, sorted(intervals, key=lambda item: (item.start, item.end)))
        )

    records: list[AudioVisualizationRecord] = []
    for audio_path, audio_annotations in sorted(annotations_by_audio.items(), key=lambda item: item[0].name):
        panels = [
            make_panel(
                tags=["VAD", *([model_name] if model_name else []), format_optional_float_tag("threshold", threshold)],
                note=display_relative(json_path),
                overlay_intervals=intervals,
                use_subtle_background=True,
            )
            for json_path, model_name, threshold, intervals in sorted(
                audio_annotations,
                key=lambda item: (item[2] is None, item[2] or 0.0, item[0].as_posix()),
            )
        ]
        records.append(make_record(audio_path, panels, display_relative(audio_path)))

    return sort_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VAD visualization.")
    parser.add_argument("--vad-json", type=Path, required=True, help="Root directory to recursively scan for VAD JSON files and matching audio.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(args.vad_json)
    run_visualization("VAD Visualization", records)


if __name__ == "__main__":
    main()

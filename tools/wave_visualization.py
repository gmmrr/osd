#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from utils.visualization import (
    AudioVisualizationRecord,
    display_relative,
    find_wav_paths,
    make_panel,
    make_record,
    sort_records,
    run_visualization,
)


def build_records(input_dir: Path) -> list[AudioVisualizationRecord]:
    records: list[AudioVisualizationRecord] = []
    for audio_path in find_wav_paths(input_dir):
        panel = make_panel(
            tags=["Wave"],
            use_subtle_background=True,
        )
        records.append(make_record(audio_path, [panel], display_relative(audio_path)))
    return sort_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave visualization.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing WAV files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(args.input_dir)
    run_visualization("Wave Visualization", records)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from utils.visualization import (
    AudioVisualizationRecord,
    SpeakerCountInterval,
    SpeakerInterval,
    audio_duration,
    display_relative,
    find_wav_paths,
    load_rttm_segments,
    make_panel,
    make_record,
    sort_records,
    run_visualization,
    sort_speakers,
)


def build_speaker_count_intervals(
    speaker_segments: list[SpeakerInterval],
    duration: float,
) -> tuple[list[SpeakerCountInterval], int]:
    events: list[tuple[float, int, str]] = []
    for segment in speaker_segments:
        start = max(0.0, segment.start)
        end = min(duration, segment.end)
        if end > start:
            events.append((start, 1, segment.speaker))
            events.append((end, -1, segment.speaker))
    events.sort(key=lambda item: item[0])

    active: Counter[str] = Counter()
    intervals: list[SpeakerCountInterval] = []
    cursor = 0.0
    event_index = 0

    while event_index < len(events):
        event_time = events[event_index][0]
        if event_time > cursor:
            intervals.append(SpeakerCountInterval(cursor, event_time, len(active)))
        while event_index < len(events) and events[event_index][0] == event_time:
            _, delta, speaker = events[event_index]
            active[speaker] += delta
            if active[speaker] <= 0:
                del active[speaker]
            event_index += 1
        cursor = event_time

    if cursor < duration:
        intervals.append(SpeakerCountInterval(cursor, duration, len(active)))

    merged: list[SpeakerCountInterval] = []
    for interval in intervals:
        if merged and merged[-1].count == interval.count:
            previous = merged[-1]
            merged[-1] = SpeakerCountInterval(previous.start, interval.end, interval.count)
        else:
            merged.append(interval)

    return merged, max((interval.count for interval in merged), default=0)


def build_records(dataset_root: Path) -> list[AudioVisualizationRecord]:
    rttm_dir = dataset_root / "rttm"
    segments_by_uri = load_rttm_segments(dataset_root)
    records: list[AudioVisualizationRecord] = []

    for audio_path in find_wav_paths(dataset_root, prefer_audio_dir=True):
        speaker_segments = segments_by_uri.get(audio_path.stem, [])
        speakers = sort_speakers(segment.speaker for segment in speaker_segments)
        count_intervals, max_concurrent_speakers = build_speaker_count_intervals(
            speaker_segments,
            audio_duration(audio_path),
        )

        panel = make_panel(
            tags=[
                "Mix",
                f"segments={len(speaker_segments)}",
                f"speakers={len(speakers)}",
                f"max speaker per frame={max_concurrent_speakers}",
            ],
            note=display_relative(rttm_dir),
            colored_speakers=speaker_segments,
            use_subtle_background=True,
            speakers=speakers,
            speaker_count_intervals=count_intervals,
            max_speaker_count=max_concurrent_speakers,
        )
        records.append(make_record(audio_path, [panel], display_relative(audio_path)))

    return sort_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix visualization.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Dataset root containing audio/ and rttm/.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(args.input_dir)
    run_visualization("Mix Visualization", records)


if __name__ == "__main__":
    main()

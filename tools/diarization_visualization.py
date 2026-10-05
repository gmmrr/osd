#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from utils.visualization import (
    AudioVisualizationRecord,
    SpeakerInterval,
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
class DiarizationAnnotation:
    json_path: Path
    intervals: list[SpeakerInterval]


def first_shared_single_speaker(
    prediction_segments: list[SpeakerInterval],
    ground_truth_segments: list[SpeakerInterval],
) -> tuple[str, str] | None:
    """Return the prediction/GT speakers at the first shared single-speaker frame."""
    boundaries = sorted(
        {
            time
            for segment in (*prediction_segments, *ground_truth_segments)
            for time in (segment.start, segment.end)
        }
    )
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = start + (end - start) / 2.0
        prediction_active = {
            segment.speaker
            for segment in prediction_segments
            if segment.start <= midpoint < segment.end
        }
        ground_truth_active = {
            segment.speaker
            for segment in ground_truth_segments
            if segment.start <= midpoint < segment.end
        }
        if len(prediction_active) == 1 and len(ground_truth_active) == 1:
            return next(iter(prediction_active)), next(iter(ground_truth_active))
    return None


def align_prediction_speakers(
    prediction_speakers: list[str],
    prediction_segments: list[SpeakerInterval],
    ground_truth_speakers: list[str],
    ground_truth_segments: list[SpeakerInterval],
) -> list[str]:
    """Align prediction color/lane order to GT using the first solo frame."""
    match = first_shared_single_speaker(prediction_segments, ground_truth_segments)
    if match is None:
        return prediction_speakers

    prediction_speaker, ground_truth_speaker = match
    ground_truth_index = ground_truth_speakers.index(ground_truth_speaker)
    if ground_truth_index >= len(prediction_speakers):
        return prediction_speakers

    aligned: list[str | None] = [None] * len(prediction_speakers)
    aligned[ground_truth_index] = prediction_speaker
    remaining = iter(
        speaker for speaker in prediction_speakers if speaker != prediction_speaker
    )
    for index, speaker in enumerate(aligned):
        if speaker is None:
            aligned[index] = next(remaining)
    return [speaker for speaker in aligned if speaker is not None]


def load_diarization_annotations(path: Path) -> dict[str, DiarizationAnnotation]:
    json_paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not json_paths or not path.exists():
        raise FileNotFoundError(f"No diarization JSON found under: {path}")

    annotations: dict[str, DiarizationAnnotation] = {}
    for json_path in json_paths:
        data = load_json_object(json_path)
        uri = str(data.get("uri") or json_path.stem)
        intervals: list[SpeakerInterval] = []
        for segment in data.get("segments", []):
            try:
                speaker = str(segment["speaker"])
                start = float(segment["start"])
                end = float(segment["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                intervals.append(SpeakerInterval(speaker=speaker, start=start, end=end))

        annotations[uri] = DiarizationAnnotation(
            json_path=json_path.resolve(),
            intervals=sorted(intervals, key=lambda item: (item.start, item.end, item.speaker)),
        )
    return annotations


def build_records(
    audio_root: Path,
    diarization_json_root: Path,
    ground_truth_rttm_dir: Path | None = None,
) -> list[AudioVisualizationRecord]:
    diarization_by_uri = load_diarization_annotations(diarization_json_root)
    ground_truth_by_uri = (
        load_rttm_directory(ground_truth_rttm_dir)
        if ground_truth_rttm_dir is not None
        else None
    )
    records: list[AudioVisualizationRecord] = []

    for audio_path in find_wav_paths(audio_root, prefer_audio_dir=True):
        annotation = diarization_by_uri.get(audio_path.stem)
        if annotation is None:
            continue

        duration = audio_duration(audio_path)
        prediction_segments = annotation.intervals
        prediction_speakers = sort_speakers(segment.speaker for segment in prediction_segments)
        prediction_counts, max_prediction_count = build_speaker_count_intervals(
            prediction_segments,
            duration,
        )

        panels = []
        if ground_truth_by_uri is not None:
            ground_truth_segments = ground_truth_by_uri.get(audio_path.stem, [])
            ground_truth_speakers = sort_speakers(segment.speaker for segment in ground_truth_segments)
            ground_truth_counts, max_ground_truth_count = build_speaker_count_intervals(
                ground_truth_segments,
                duration,
            )
            prediction_speakers = align_prediction_speakers(
                prediction_speakers,
                prediction_segments,
                ground_truth_speakers,
                ground_truth_segments,
            )
            shared_max_count = max(max_ground_truth_count, max_prediction_count)
            panels.append(
                make_panel(
                    tags=[
                        "Ground Truth",
                        f"segments={len(ground_truth_segments)}",
                        f"speakers={len(ground_truth_speakers)}",
                        f"max speakers per frame={max_ground_truth_count}",
                    ],
                    note=display_relative(ground_truth_rttm_dir),
                    colored_speakers=ground_truth_segments,
                    use_subtle_background=True,
                    speakers=ground_truth_speakers,
                    speaker_count_intervals=ground_truth_counts,
                    max_speaker_count=shared_max_count,
                )
            )
        else:
            shared_max_count = max_prediction_count

        panels.append(
            make_panel(
                tags=[
                    "Diarization Prediction",
                    f"segments={len(prediction_segments)}",
                    f"speakers={len(prediction_speakers)}",
                    f"max speakers per frame={max_prediction_count}",
                ],
                note=display_relative(annotation.json_path),
                colored_speakers=prediction_segments,
                use_subtle_background=True,
                speakers=prediction_speakers,
                speaker_count_intervals=prediction_counts,
                max_speaker_count=shared_max_count,
            )
        )
        records.append(make_record(audio_path, panels, display_relative(audio_path)))

    if not records:
        raise RuntimeError("No shared URIs between audio files and diarization JSON.")
    return sort_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize diarization predictions for the same audio.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing the original audio files.")
    parser.add_argument(
        "--diarization-json",
        type=Path,
        required=True,
        help="Diarization JSON file or directory containing JSON files.",
    )
    parser.add_argument(
        "--ground-truth",
        "--rttm-dir",
        dest="ground_truth",
        type=Path,
        help="Optional directory containing ground-truth RTTM files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(args.audio_dir, args.diarization_json, args.ground_truth)
    title = "Diarization Demo Visualization" if args.ground_truth is not None else "Diarization Visualization"
    run_visualization(title, records)


if __name__ == "__main__":
    main()

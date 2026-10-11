#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scipy.optimize import linear_sum_assignment

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


def _speaker_timelines(
    speakers: list[str],
    segments: list[SpeakerInterval],
) -> dict[str, list[tuple[float, float]]]:
    """Build merged, non-overlapping intervals for every speaker."""
    timelines: dict[str, list[tuple[float, float]]] = {speaker: [] for speaker in speakers}
    for speaker in speakers:
        speaker_segments = sorted(
            (segment.start, segment.end)
            for segment in segments
            if segment.speaker == speaker and segment.end > segment.start
        )
        for start, end in speaker_segments:
            if timelines[speaker] and start <= timelines[speaker][-1][1]:
                previous_start, previous_end = timelines[speaker][-1]
                timelines[speaker][-1] = (previous_start, max(previous_end, end))
            else:
                timelines[speaker].append((start, end))
    return timelines


def _overlap_duration(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> float:
    """Return total intersection duration of two merged timelines."""
    total = 0.0
    first_index = 0
    second_index = 0
    while first_index < len(first) and second_index < len(second):
        first_start, first_end = first[first_index]
        second_start, second_end = second[second_index]
        total += max(0.0, min(first_end, second_end) - max(first_start, second_start))
        if first_end <= second_end:
            first_index += 1
        else:
            second_index += 1
    return total


def optimal_speaker_alignment(
    prediction_speakers: list[str],
    prediction_segments: list[SpeakerInterval],
    ground_truth_speakers: list[str],
    ground_truth_segments: list[SpeakerInterval],
) -> dict[str, str]:
    """Find the one-to-one speaker mapping with maximum total time overlap."""
    if not prediction_speakers or not ground_truth_speakers:
        return {}

    prediction_timelines = _speaker_timelines(prediction_speakers, prediction_segments)
    ground_truth_timelines = _speaker_timelines(ground_truth_speakers, ground_truth_segments)
    overlap = [
        [
            _overlap_duration(
                prediction_timelines[prediction_speaker],
                ground_truth_timelines[ground_truth_speaker],
            )
            for ground_truth_speaker in ground_truth_speakers
        ]
        for prediction_speaker in prediction_speakers
    ]

    prediction_indices, ground_truth_indices = linear_sum_assignment(
        [[-duration for duration in row] for row in overlap]
    )
    return {
        prediction_speakers[prediction_index]: ground_truth_speakers[ground_truth_index]
        for prediction_index, ground_truth_index in zip(
            prediction_indices, ground_truth_indices
        )
        if overlap[prediction_index][ground_truth_index] > 0.0
    }


def align_prediction_speakers(
    prediction_speakers: list[str],
    prediction_segments: list[SpeakerInterval],
    ground_truth_speakers: list[str],
    ground_truth_segments: list[SpeakerInterval],
) -> list[str]:
    """Align prediction color/lane order using maximum-overlap assignment."""
    mapping = optimal_speaker_alignment(
        prediction_speakers,
        prediction_segments,
        ground_truth_speakers,
        ground_truth_segments,
    )
    if not mapping:
        return prediction_speakers

    aligned: list[str | None] = [None] * len(prediction_speakers)
    remaining: list[str] = []
    ground_truth_indices = {
        speaker: index for index, speaker in enumerate(ground_truth_speakers)
    }
    for prediction_speaker in prediction_speakers:
        ground_truth_speaker = mapping.get(prediction_speaker)
        ground_truth_index = (
            ground_truth_indices[ground_truth_speaker]
            if ground_truth_speaker is not None
            else len(aligned)
        )
        if ground_truth_index < len(aligned) and aligned[ground_truth_index] is None:
            aligned[ground_truth_index] = prediction_speaker
        else:
            remaining.append(prediction_speaker)

    remaining_iter = iter(remaining)
    for index, speaker in enumerate(aligned):
        if speaker is None:
            aligned[index] = next(remaining_iter)
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

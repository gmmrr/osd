#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate mixtures with explicit frame-level speaker-count proportions."""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import (
    STEP_1_AUDIO_SAMPLE_RATE,
    STEP_4_GAIN_DB_RANGE,
    STEP_4_MAX_BUILD_TRIALS,
    STEP_4_MAX_MIXTURE_DURATION,
    STEP_4_MIN_MIXTURE_DURATION,
    STEP_4_MIN_SEGMENT_DURATION,
    STEP_4_SEED,
)
from pipeline.step_4_metadata import (
    ActiveSegment,
    SpeakerGroup,
    build_source,
    collect_groups,
    format_usage_histogram,
    weighted_choice_without_replacement,
    write_json,
)
from pipeline.utils.progress import write_progress


def natural_speaker_count_ratios(max_count: int) -> list[float]:
    """Approximate natural conversation: mostly one speaker, then silence, with rapidly decreasing overlap."""
    weights = [0.15, 0.75]
    weights.extend(max(0.08 * 0.25 ** (count - 2), 0.001) for count in range(2, max_count + 1))
    total = sum(weights[: max_count + 1])
    return [value / total for value in weights[: max_count + 1]]


def normalize_count_ratios(values: list[float] | None, max_count: int) -> list[float]:
    ratios = [1.0] * (max_count + 1) if values is None else [float(value) for value in values]
    if len(ratios) != max_count + 1:
        raise ValueError(f"Expected {max_count + 1} speaker-count ratios for counts 0 through {max_count}.")
    if any(value < 0.0 for value in ratios) or sum(ratios) <= 0.0:
        raise ValueError("Speaker-count ratios must be non-negative and contain at least one positive value.")
    total = sum(ratios)
    return [value / total for value in ratios]


def resolve_count_ratios(values: list[str] | None, max_count: int) -> list[float]:
    if values is not None and len(values) == 1 and values[0].lower() == "natural":
        return natural_speaker_count_ratios(max_count)
    if values is not None and any(value.lower() == "natural" for value in values):
        raise ValueError("natural must be used by itself.")
    try:
        numeric_values = None if values is None else [float(value) for value in values]
    except ValueError as error:
        raise ValueError("Ratios must be numbers or the single word natural.") from error
    return normalize_count_ratios(numeric_values, max_count)


def merge_speaker_groups(groups: list[SpeakerGroup]) -> list[SpeakerGroup]:
    segments_by_speaker: dict[str, list[ActiveSegment]] = {}
    for group in groups:
        speaker = next((segment.speaker for segment in group.segments if segment.speaker not in (None, "")), None)
        key = str(speaker) if speaker is not None else group.key
        segments_by_speaker.setdefault(key, []).extend(group.segments)

    return [
        SpeakerGroup(
            key=key,
            segments=sorted(segments, key=lambda segment: (segment.audio_derivative, segment.orig_start)),
            total_active_duration=sum(segment.duration for segment in segments),
        )
        for key, segments in sorted(segments_by_speaker.items())
    ]


def remaining_duration(queue: list[ActiveSegment]) -> float:
    return sum(segment.duration for segment in queue)


def choose_speakers(
    queues: dict[str, list[ActiveSegment]],
    count: int,
    required_duration: float,
    rng: random.Random,
) -> list[str]:
    candidates = [
        SpeakerGroup(key=key, segments=[], total_active_duration=remaining_duration(queue))
        for key, queue in queues.items()
        if remaining_duration(queue) + 1e-9 >= required_duration
    ]
    if len(candidates) < count:
        return []
    return [group.key for group in weighted_choice_without_replacement(candidates, count, rng)]


def fill_speaker_region(
    queue: list[ActiveSegment],
    speaker: str,
    start: float,
    end: float,
    sources: list[dict[str, Any]],
    rng: random.Random,
) -> bool:
    cursor = start
    gain_db = rng.uniform(*STEP_4_GAIN_DB_RANGE)
    while cursor < end - 1e-9:
        if not queue:
            return False

        segment = queue.pop()
        duration = min(segment.duration, end - cursor)
        piece = ActiveSegment(
            source_id=segment.source_id,
            audio=segment.audio,
            audio_derivative=segment.audio_derivative,
            speaker=speaker,
            segment_index=segment.segment_index,
            orig_start=segment.orig_start,
            orig_end=segment.orig_start + duration,
        )
        source_end = cursor + duration
        sources.append(build_source(piece, speaker, len(sources) + 1, cursor, source_end, gain_db))

        if duration < segment.duration - 1e-9:
            queue.append(
                ActiveSegment(
                    source_id=segment.source_id,
                    audio=segment.audio,
                    audio_derivative=segment.audio_derivative,
                    speaker=speaker,
                    segment_index=segment.segment_index,
                    orig_start=segment.orig_start + duration,
                    orig_end=segment.orig_end,
                )
            )
        cursor = source_end
    return True


def speaker_count_segments(sources: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    events: list[tuple[float, int, str, str]] = []
    for source in sources:
        events.append((float(source["mix_start"]), 1, str(source["speaker"]), str(source["source"])))
        events.append((float(source["mix_end"]), 0, str(source["speaker"]), str(source["source"])))
    events.sort(key=lambda item: (item[0], item[1]))

    active_speakers: Counter[str] = Counter()
    active_sources: set[str] = set()
    intervals: list[dict[str, Any]] = []
    previous = 0.0

    for time, kind, speaker, source_id in events:
        if time > previous + 1e-9:
            intervals.append(
                {
                    "start": round(previous, 6),
                    "end": round(time, 6),
                    "duration": round(time - previous, 6),
                    "count": len(active_speakers),
                    "speakers": sorted(active_speakers),
                    "sources": sorted(active_sources),
                }
            )
        if kind == 0:
            active_speakers[speaker] -= 1
            if active_speakers[speaker] <= 0:
                active_speakers.pop(speaker, None)
            active_sources.discard(source_id)
        else:
            active_speakers[speaker] += 1
            active_sources.add(source_id)
        previous = time

    if duration > previous + 1e-9:
        intervals.append(
            {
                "start": round(previous, 6),
                "end": round(duration, 6),
                "duration": round(duration - previous, 6),
                "count": len(active_speakers),
                "speakers": sorted(active_speakers),
                "sources": sorted(active_sources),
            }
        )
    return intervals


def build_candidate_mixture(
    input_dir: Path,
    uri: str,
    groups: list[SpeakerGroup],
    rng: random.Random,
    run_uuid: str,
    count_ratios: list[float],
    min_mixture_duration: float,
    max_mixture_duration: float,
    max_speakers: int | None,
    max_speakers_per_frame: int,
) -> dict[str, Any] | None:
    max_count = max_speakers_per_frame
    required_speakers = max(count for count, ratio in enumerate(count_ratios) if ratio > 0.0)
    pool_size = required_speakers if max_speakers is None else min(max_speakers, len(groups))
    if pool_size < required_speakers:
        return None

    selected = weighted_choice_without_replacement(groups, pool_size, rng)
    queues = {group.key: list(group.segments) for group in selected}
    for queue in queues.values():
        rng.shuffle(queue)

    duration = rng.uniform(min_mixture_duration, max_mixture_duration) / 1000.0
    ordered_counts = [count for count, ratio in enumerate(count_ratios) if ratio > 0.0]
    rng.shuffle(ordered_counts)

    regions: list[tuple[int, float, float]] = []
    cursor = 0.0
    for index, count in enumerate(ordered_counts):
        end = duration if index + 1 == len(ordered_counts) else cursor + duration * count_ratios[count]
        regions.append((count, cursor, end))
        cursor = end

    sources: list[dict[str, Any]] = []
    for count, start, end in regions:
        if count == 0:
            continue
        speakers = choose_speakers(queues, count, end - start, rng)
        if len(speakers) != count:
            return None
        for speaker in speakers:
            if not fill_speaker_region(queues[speaker], speaker, start, end, sources, rng):
                return None

    count_segments = speaker_count_segments(sources, duration)
    if max((int(segment["count"]) for segment in count_segments), default=0) > max_count:
        return None

    actual_durations = [0.0] * (max_count + 1)
    for segment in count_segments:
        actual_durations[int(segment["count"])] += float(segment["duration"])
    actual_ratios = [value / duration for value in actual_durations]
    speakers = list(dict.fromkeys(str(source["speaker"]) for source in sources))
    global_overlap = [segment for segment in count_segments if int(segment["count"]) >= 2]

    return {
        "uri": uri,
        "uuid": run_uuid,
        "input": str(input_dir),
        "duration": round(duration, 6),
        "sample_rate": STEP_1_AUDIO_SAMPLE_RATE,
        "speaker_count_ratios_target": [round(value, 6) for value in count_ratios],
        "speaker_count_ratios_actual": [round(value, 6) for value in actual_ratios],
        "pre_silence": 0.0,
        "post_silence": 0.0,
        "max_speakers": max_speakers,
        "max_speakers_per_frame": max_speakers_per_frame,
        "speakers": speakers,
        "speaker_count": len(speakers),
        "source_count": len(sources),
        "overlap_count": len(global_overlap),
        "global_overlap_count": len(global_overlap),
        "sources": sources,
        "overlap": [],
        "global_overlap": global_overlap,
        "speaker_count_segments": count_segments,
    }


def run_metadata_dir(args: argparse.Namespace) -> None:
    groups = merge_speaker_groups(collect_groups(args.vad_dir, args.min_segment_duration))
    max_count = args.max_speakers_per_frame
    if max_count < 3:
        raise ValueError("Alternative Step 4 requires --max-speakers-per-frame of at least 3.")
    count_ratios = resolve_count_ratios(args.max_speakers_per_frame_ratio, max_count)
    required_speakers = max(count for count, ratio in enumerate(count_ratios) if ratio > 0.0)
    max_speakers = max(args.max_speakers or required_speakers, required_speakers)
    if len(groups) < required_speakers:
        raise RuntimeError(f"Need at least {required_speakers} distinct speaker groups.")

    rng = random.Random(args.seed)
    run_uuid = uuid.uuid4().hex[:8]
    mixtures: list[dict[str, Any]] = []
    attempts = 0

    print(f"🚀 Generate count-balanced metadata in '{args.vad_dir}'")
    print(f"   • Speaker groups: {len(groups)}")
    print(f"   • Target mixtures: {args.n_mixtures}")
    print(f"   • Count ratios: {', '.join(f'{count}={ratio:.4f}' for count, ratio in enumerate(count_ratios))}")

    while len(mixtures) < args.n_mixtures:
        attempts += 1
        if attempts > args.max_build_trials:
            raise RuntimeError(f"Only generated {len(mixtures)} mixtures after {args.max_build_trials} attempts.")
        record = build_candidate_mixture(
            input_dir=args.vad_dir,
            uri=f"{args.prefix}_{len(mixtures):08d}",
            groups=groups,
            rng=rng,
            run_uuid=run_uuid,
            count_ratios=count_ratios,
            min_mixture_duration=args.min_mixture_duration,
            max_mixture_duration=args.max_mixture_duration,
            max_speakers=max_speakers,
            max_speakers_per_frame=max_count,
        )
        if record is not None:
            mixtures.append(record)
            write_progress(len(mixtures), args.n_mixtures)

    write_json(mixtures, args.out, force=args.force)
    usage_counts = Counter(
        name for mixture in mixtures for name in {Path(source["audio"]).stem for source in mixture["sources"]}
    )
    print()
    print(f"   • Attempts: {attempts}")
    for line in format_usage_histogram(usage_counts):
        print(line)
    print(f"   • Output: {args.out}")
    print("✅ Count-balanced metadata completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mixtures with explicit frame-level speaker-count proportions.")
    parser.add_argument("--input-dir", dest="vad_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-mixtures", type=int, required=True)
    parser.add_argument("--min-segment-duration", type=float, default=STEP_4_MIN_SEGMENT_DURATION)
    parser.add_argument("--min-mixture-duration", type=float, default=STEP_4_MIN_MIXTURE_DURATION)
    parser.add_argument("--max-mixture-duration", type=float, default=STEP_4_MAX_MIXTURE_DURATION)
    parser.add_argument("--seed", type=int, default=STEP_4_SEED)
    parser.add_argument("--prefix", type=str, default="mix")
    parser.add_argument("--max-build-trials", type=int, default=STEP_4_MAX_BUILD_TRIALS)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--max-speakers-per-frame", type=int, default=3)
    parser.add_argument("--max-speakers-per-frame-ratio", nargs="+")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_metadata_dir(parse_args())


if __name__ == "__main__":
    main()

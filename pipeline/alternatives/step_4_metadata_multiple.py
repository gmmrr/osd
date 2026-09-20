#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate N-speaker mixtures from exact frame-level overlap quotas."""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import (
    STEP_1_AUDIO_SAMPLE_RATE,
    STEP_4_ALT_FRAME_RESOLUTION,
    STEP_4_ALT_MAX_EVENT_DURATION,
    STEP_4_ALT_MAX_SESSION_DURATION,
    STEP_4_ALT_MIN_DURATION_RATIO_THRESHOLD,
    STEP_4_ALT_MIN_EVENT_DURATION,
    STEP_4_ALT_MIN_OVERLAP_ORDER_DURATION,
    STEP_4_ALT_RATIO_TOLERANCE,
    STEP_4_GAIN_DB_RANGE,
    STEP_4_MAX_MIXTURE_DURATION,
    STEP_4_MIN_MIXTURE_DURATION,
    STEP_4_MIN_SEGMENT_DURATION,
    STEP_4_SAME_SPEAKER_GAP,
    STEP_4_SEED,
)
from pipeline.step_4_metadata import (
    ActiveSegment,
    SpeakerGroup,
    build_source,
    collect_groups,
    weighted_choice_without_replacement,
    write_json,
)
from pipeline.utils.progress import write_progress

def normalize_ratio_map(overlap_ratios: Mapping[int, float], n_speakers: int) -> dict[int, float]:
    """Validate and normalize a complete 0..N frame-ratio mapping."""
    ratios = {int(order): float(ratio) for order, ratio in overlap_ratios.items()}
    invalid_orders = [order for order in ratios if not 0 <= order <= n_speakers]
    if invalid_orders:
        raise ValueError(f"Overlap orders must be between 0 and {n_speakers}, got {sorted(invalid_orders)}.")
    if any(ratio < 0.0 for ratio in ratios.values()):
        raise ValueError("Overlap ratios must be non-negative.")
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=STEP_4_ALT_RATIO_TOLERANCE):
        raise ValueError(f"Overlap ratios must sum to 1.0, got {total:.12g}.")
    return {order: ratio / total for order, ratio in ratios.items()}


def natural_speaker_count_ratios(max_count: int) -> list[float]:
    """Return non-zero natural-conversation weights for exactly 1..max_count active speakers."""
    if max_count < 1:
        raise ValueError("max_count must be positive.")
    weights = [0.90]
    if max_count >= 2:
        weights.append(0.08)
    weights.extend(max(0.015 * 0.25 ** (count - 3), 0.0001) for count in range(3, max_count + 1))
    total = sum(weights)
    return [value / total for value in weights]


def normalize_count_ratios(values: Sequence[float] | None, max_count: int) -> list[float]:
    """Normalize non-negative overlap weights for active-speaker orders 1..max_count."""
    if max_count < 1:
        raise ValueError("max_count must be positive.")
    ratios = [1.0 / max_count] * max_count if values is None else [float(value) for value in values]
    if len(ratios) != max_count:
        raise ValueError(f"Expected {max_count} ratios for overlap orders 1 through {max_count}.")
    if any(value < 0.0 for value in ratios) or sum(ratios) <= 0.0:
        raise ValueError("Overlap ratios must be non-negative and contain at least one positive value.")
    total = sum(ratios)
    return [value / total for value in ratios]


def calculate_required_session_duration(
    overlap_ratios: Mapping[int, float],
    minimum_order_duration: float,
    minimum_ratio_threshold: float = STEP_4_ALT_MIN_DURATION_RATIO_THRESHOLD,
    frame_resolution: float = STEP_4_ALT_FRAME_RESOLUTION,
) -> float:
    """Return the required duration in milliseconds for sufficiently frequent overlap orders."""
    if minimum_order_duration <= 0.0:
        raise ValueError("minimum_order_duration must be positive.")
    if not 0.0 <= minimum_ratio_threshold <= 1.0:
        raise ValueError("minimum_ratio_threshold must be between 0 and 1.")
    ratios = {int(order): float(ratio) for order, ratio in overlap_ratios.items()}
    ratios = normalize_ratio_map(ratios, max(ratios, default=0))
    active_ratios = [ratio for order, ratio in ratios.items() if order > 0 and ratio > 0.0]
    if not active_ratios:
        raise ValueError("At least one active overlap order must have a positive ratio.")
    eligible_ratios = [ratio for ratio in active_ratios if ratio >= minimum_ratio_threshold]
    if not eligible_ratios:
        return 0.0
    minimum_order_frames = math.ceil(minimum_order_duration / frame_resolution)
    required_frames = max(math.ceil(minimum_order_frames / ratio) for ratio in eligible_ratios)
    return required_frames * frame_resolution


def calculate_session_duration_range(
    overlap_ratios: Mapping[int, float],
    configured_min_duration: float,
    configured_max_duration: float,
    minimum_order_duration: float,
    minimum_ratio_threshold: float,
    max_session_duration: float,
    frame_resolution: float,
) -> tuple[float, float, float]:
    """Return required, effective minimum, and effective maximum durations in milliseconds."""
    if max_session_duration <= 0.0:
        raise ValueError("max_session_duration must be positive.")
    if configured_min_duration > max_session_duration:
        raise ValueError("configured_min_duration cannot exceed max_session_duration.")
    required_duration = calculate_required_session_duration(
        overlap_ratios,
        minimum_order_duration,
        minimum_ratio_threshold,
        frame_resolution,
    )
    required_frames = round(required_duration / frame_resolution)
    configured_min_frames = math.ceil(configured_min_duration / frame_resolution)
    configured_max_frames = math.floor(configured_max_duration / frame_resolution)
    maximum_frames = math.floor(max_session_duration / frame_resolution)
    min_frames = min(max(configured_min_frames, required_frames), maximum_frames)
    max_frames = min(max(configured_max_frames, min_frames), maximum_frames)
    return required_duration, min_frames * frame_resolution, max_frames * frame_resolution


def resolve_count_ratios(values: Sequence[str | float] | None, max_count: int) -> list[float]:
    if values is not None and len(values) == 1 and str(values[0]).lower() == "natural":
        return natural_speaker_count_ratios(max_count)
    numeric_values = None if values is None else [float(value) for value in values]
    return normalize_count_ratios(numeric_values, max_count)


def allocate_frame_quotas(
    n_speakers: int,
    duration: float,
    overlap_ratios: Mapping[int, float],
    frame_resolution: float,
) -> tuple[int, dict[int, int]]:
    """Allocate exact quotas from millisecond durations with the Hamilton method."""
    if n_speakers < 1:
        raise ValueError("n_speakers must be positive.")
    if duration <= 0.0 or frame_resolution <= 0.0:
        raise ValueError("duration and frame_resolution must be positive.")
    ratios = normalize_ratio_map(overlap_ratios, n_speakers)

    total_frames = round(duration / frame_resolution)
    if total_frames <= 0:
        raise ValueError("duration is shorter than one frame.")

    raw = {order: total_frames * ratio for order, ratio in ratios.items()}
    quota = {order: math.floor(value) for order, value in raw.items()}
    remaining = total_frames - sum(quota.values())
    remainder_order = sorted(raw, key=lambda order: (-(raw[order] - quota[order]), order))
    for order in remainder_order[:remaining]:
        quota[order] += 1

    return total_frames, quota


def _split_frame_quota(
    quota: int,
    min_frames: int,
    max_frames: int,
    rng: random.Random,
) -> list[int]:
    """Split one quota without creating a short remainder unless the whole quota is short."""
    chunks: list[int] = []
    remaining = quota
    while remaining > max_frames:
        upper = min(max_frames, remaining - min_frames)
        duration = rng.randint(min_frames, upper) if upper >= min_frames else max_frames
        chunks.append(duration)
        remaining -= duration
    if remaining:
        chunks.append(remaining)
    rng.shuffle(chunks)
    return chunks


def generate_count_trajectory(
    frame_quota: Mapping[int, int],
    frame_resolution: float,
    min_event_duration: float,
    max_event_duration: float,
    rng: random.Random,
) -> list[dict[str, int]]:
    """Create an exact-quota count random walk from millisecond event settings."""
    if min_event_duration <= 0.0 or max_event_duration < min_event_duration:
        raise ValueError("Event durations must satisfy 0 < min_event_duration <= max_event_duration.")
    min_frames = max(1, math.ceil(min_event_duration / frame_resolution))
    max_frames = math.floor(max_event_duration / frame_resolution)
    if max_frames < min_frames:
        raise ValueError("Event duration range does not contain a complete frame.")

    positive_orders = [order for order, quota in sorted(frame_quota.items()) if quota > 0]
    if positive_orders and (
        positive_orders[0] > 1
        or positive_orders != list(range(positive_orders[0], positive_orders[-1] + 1))
    ):
        raise ValueError(
            "Non-contiguous active-speaker support cannot be realized with "
            "single-speaker transitions while preserving exact frame quotas."
        )
    if not positive_orders:
        return []

    chunks = {
        order: _split_frame_quota(int(frame_quota[order]), min_frames, max_frames, rng)
        for order in positive_orders
    }
    initial_state = positive_orders[0]
    initial_counts = tuple(len(chunks[order]) for order in positive_orders)
    initial_counts = tuple(
        count - 1 if order == initial_state else count
        for order, count in zip(positive_orders, initial_counts)
    )
    failed: set[tuple[int, tuple[int, ...]]] = set()

    def weighted_candidates(current: int, counts: tuple[int, ...]) -> list[int]:
        candidates = [
            order
            for index, order in enumerate(positive_orders)
            if counts[index] and abs(order - current) <= 1
        ]
        ordered: list[int] = []
        while candidates:
            weights = [
                sum(chunks[order][: counts[positive_orders.index(order)]])
                for order in candidates
            ]
            selected = rng.choices(candidates, weights=weights, k=1)[0]
            ordered.append(selected)
            candidates.remove(selected)
        return ordered

    def find_path(current: int, counts: tuple[int, ...]) -> list[int] | None:
        if not any(counts):
            return []
        key = (current, counts)
        if key in failed:
            return None

        for next_state in weighted_candidates(current, counts):
            index = positive_orders.index(next_state)
            next_counts = list(counts)
            next_counts[index] -= 1
            remaining_states = {
                order for order, count in zip(positive_orders, next_counts) if count
            }
            reachable = remaining_states | {next_state}
            if reachable and any(
                order not in reachable
                for order in range(min(reachable), max(reachable) + 1)
            ):
                continue
            suffix = find_path(next_state, tuple(next_counts))
            if suffix is not None:
                return [next_state, *suffix]

        failed.add(key)
        return None

    state_path = [initial_state]
    suffix = find_path(initial_state, initial_counts)
    if suffix is None:
        raise RuntimeError("Unable to realize exact frame quotas as a contiguous count trajectory.")
    state_path.extend(suffix)

    remaining_chunks = {order: list(values) for order, values in chunks.items()}
    return [
        {"num_active": order, "duration_frames": remaining_chunks[order].pop()}
        for order in state_path
    ]


def assign_speaker_transitions(
    events: Sequence[Mapping[str, int]],
    speakers: Sequence[str],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Assign persistent speaker sets with one seeded, balanced join or leave per transition."""
    speaker_list = list(speakers)
    if len(set(speaker_list)) != len(speaker_list):
        raise ValueError("Session speakers must be unique.")
    speaker_active_frames: Counter[str] = Counter()
    speaker_turn_count: Counter[str] = Counter()
    active: set[str] = set()
    assigned: list[dict[str, Any]] = []

    for event in events:
        order = int(event["num_active"])
        duration_frames = int(event["duration_frames"])
        if not active and order:
            for _ in range(order):
                inactive = [speaker for speaker in speaker_list if speaker not in active]
                best = min((speaker_active_frames[speaker], speaker_turn_count[speaker]) for speaker in inactive)
                candidates = [
                    speaker for speaker in inactive
                    if (speaker_active_frames[speaker], speaker_turn_count[speaker]) == best
                ]
                joined = rng.choice(candidates)
                active.add(joined)
                speaker_turn_count[joined] += 1
        elif order == len(active) + 1:
            inactive = [speaker for speaker in speaker_list if speaker not in active]
            best = min((speaker_active_frames[speaker], speaker_turn_count[speaker]) for speaker in inactive)
            candidates = [
                speaker for speaker in inactive
                if (speaker_active_frames[speaker], speaker_turn_count[speaker]) == best
            ]
            joined = rng.choice(candidates)
            active.add(joined)
            speaker_turn_count[joined] += 1
        elif order == len(active) - 1:
            worst = max((speaker_active_frames[speaker], speaker_turn_count[speaker]) for speaker in active)
            candidates = [
                speaker for speaker in active
                if (speaker_active_frames[speaker], speaker_turn_count[speaker]) == worst
            ]
            active.remove(rng.choice(candidates))
        elif order != len(active):
            raise ValueError("Speaker-count trajectory contains a transition larger than one.")

        selected = sorted(active)
        for speaker in active:
            speaker_active_frames[speaker] += duration_frames
        assigned.append({**event, "speakers": list(selected)})

    return assigned, speaker_active_frames


def build_continuous_timeline(
    events: Sequence[Mapping[str, Any]], frame_resolution: float
) -> list[dict[str, Any]]:
    """Lay events on a frame timeline and expose metadata timestamps in seconds."""
    timeline: list[dict[str, Any]] = []
    cursor = 0
    for event in events:
        duration_frames = int(event["duration_frames"])
        end_frame = cursor + duration_frames
        timeline.append(
            {
                "start_frame": cursor,
                "end_frame": end_frame,
                "duration_frames": duration_frames,
                "start": round(cursor * frame_resolution / 1000.0, 9),
                "end": round(end_frame * frame_resolution / 1000.0, 9),
                "num_active": int(event["num_active"]),
                "speakers": list(event["speakers"]),
            }
        )
        cursor = end_frame
    return timeline


def build_speaker_activity_regions(
    events: Sequence[Mapping[str, Any]],
    frame_resolution: float,
) -> list[dict[str, Any]]:
    """Merge adjacent count-state events into continuous per-speaker regions."""
    open_regions: dict[str, int] = {}
    regions: list[dict[str, Any]] = []

    for event in events:
        start_frame = int(event["start_frame"])
        active = {str(speaker) for speaker in event["speakers"]}
        for speaker in list(open_regions):
            if speaker not in active:
                end_frame = start_frame
                regions.append(
                    {
                        "speaker": speaker,
                        "start_frame": open_regions.pop(speaker),
                        "end_frame": end_frame,
                    }
                )
        for speaker in active:
            open_regions.setdefault(speaker, start_frame)

    total_frames = int(events[-1]["end_frame"]) if events else 0
    for speaker, start_frame in open_regions.items():
        regions.append(
            {
                "speaker": speaker,
                "start_frame": start_frame,
                "end_frame": total_frames,
            }
        )

    for region in regions:
        region["duration_frames"] = region["end_frame"] - region["start_frame"]
        region["start"] = round(region["start_frame"] * frame_resolution / 1000.0, 9)
        region["end"] = round(region["end_frame"] * frame_resolution / 1000.0, 9)
    return sorted(regions, key=lambda region: (region["start_frame"], region["speaker"]))


def validate_speaker_activity_regions(
    events: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
) -> None:
    """Verify positive, non-overlapping regions reconstruct every event speaker set."""
    by_speaker: dict[str, list[tuple[int, int]]] = {}
    for region in regions:
        start = int(region["start_frame"])
        end = int(region["end_frame"])
        if end <= start:
            raise ValueError("Speaker activity region must have positive duration.")
        by_speaker.setdefault(str(region["speaker"]), []).append((start, end))

    for speaker, intervals in by_speaker.items():
        intervals.sort()
        if any(previous[1] > current[0] for previous, current in zip(intervals, intervals[1:])):
            raise ValueError(f"Speaker activity regions overlap for {speaker}.")

    for event in events:
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        reconstructed = {
            speaker
            for speaker, intervals in by_speaker.items()
            if any(region_start <= start and region_end >= end for region_start, region_end in intervals)
        }
        if reconstructed != set(event["speakers"]):
            raise ValueError("Speaker activity regions do not reconstruct the event timeline.")

    event_frames: Counter[str] = Counter()
    region_frames: Counter[str] = Counter()
    for event in events:
        for speaker in event["speakers"]:
            event_frames[str(speaker)] += int(event["duration_frames"])
    for region in regions:
        region_frames[str(region["speaker"])] += int(region["end_frame"]) - int(region["start_frame"])
    if event_frames != region_frames:
        raise ValueError("Speaker activity regions do not preserve per-speaker active frames.")


def validate_timeline_metadata(metadata: Mapping[str, Any]) -> tuple[dict[int, int], dict[int, float]]:
    """Verify exact, gap-free frame quotas and return the observed distribution."""
    total_frames = int(metadata["total_frames"])
    target_quota = {int(order): int(value) for order, value in metadata["target_frame_quota"].items()}
    actual_frames: Counter[int] = Counter()
    cursor = 0
    known_speakers = set(metadata["speakers"])

    previous_event: Mapping[str, Any] | None = None
    for event in metadata["events"]:
        start_frame = int(event["start_frame"])
        end_frame = int(event["end_frame"])
        duration_frames = int(event["duration_frames"])
        num_active = int(event["num_active"])
        event_speakers = event["speakers"]
        if start_frame != cursor:
            raise ValueError(f"Timeline gap at frame {cursor}.")
        if end_frame <= start_frame or duration_frames != end_frame - start_frame:
            raise ValueError("Event duration does not match its frame range.")
        if len(event_speakers) != num_active or len(set(event_speakers)) != num_active:
            raise ValueError("Speaker assignment does not match num_active.")
        if not set(event_speakers) <= known_speakers:
            raise ValueError("Event contains an unknown speaker.")
        if previous_event is not None:
            previous_speakers = set(previous_event["speakers"])
            current_speakers = set(event_speakers)
            delta = num_active - int(previous_event["num_active"])
            if abs(delta) > 1:
                raise ValueError("Speaker-count transition is larger than one.")
            if delta == 1 and not (
                previous_speakers < current_speakers
                and len(current_speakers - previous_speakers) == 1
            ):
                raise ValueError("Join transition must preserve all active speakers and add one.")
            if delta == -1 and not (
                current_speakers < previous_speakers
                and len(previous_speakers - current_speakers) == 1
            ):
                raise ValueError("Leave transition must preserve all remaining active speakers.")
            if delta == 0 and current_speakers != previous_speakers:
                raise ValueError("Same-count transition must preserve the active speaker set.")
        actual_frames[num_active] += duration_frames
        cursor = end_frame
        previous_event = event

    actual = {order: actual_frames[order] for order in target_quota}
    if cursor != total_frames or actual != target_quota:
        raise ValueError("Generated timeline does not match its target frame quota.")
    actual_ratio = {order: actual[order] / total_frames for order in actual}
    return actual, actual_ratio


def _generate_timeline_metadata(
    speakers: Sequence[str],
    duration: float,
    overlap_ratios: Mapping[int, float],
    frame_resolution: float,
    min_event_duration: float,
    max_event_duration: float,
    rng: random.Random,
) -> dict[str, Any]:
    total_frames, frame_quota = allocate_frame_quotas(
        len(speakers), duration, overlap_ratios, frame_resolution
    )
    normalized_ratios = normalize_ratio_map(overlap_ratios, len(speakers))
    events = generate_count_trajectory(
        frame_quota, frame_resolution, min_event_duration, max_event_duration, rng
    )
    assigned, speaker_active_frames = assign_speaker_transitions(events, speakers, rng)
    timeline = build_continuous_timeline(assigned, frame_resolution)
    metadata: dict[str, Any] = {
        "duration": round(total_frames * frame_resolution / 1000.0, 9),
        "frame_resolution": frame_resolution / 1000.0,
        "total_frames": total_frames,
        "speakers": list(speakers),
        "target_ratio": {
            str(order): round(normalized_ratios[order], 12)
            for order in sorted(normalized_ratios)
        },
        "target_frame_quota": {str(order): frame_quota[order] for order in sorted(frame_quota)},
        "events": timeline,
        "speaker_active_frames": {speaker: speaker_active_frames[speaker] for speaker in speakers},
    }
    actual_frames, actual_ratio = validate_timeline_metadata(metadata)
    metadata["actual_frames"] = {str(order): value for order, value in actual_frames.items()}
    metadata["actual_ratio"] = {str(order): value for order, value in actual_ratio.items()}
    return metadata


def generate_timeline_metadata(
    speakers: Sequence[str],
    duration: float,
    overlap_ratios: Mapping[int, float],
    frame_resolution: float = STEP_4_ALT_FRAME_RESOLUTION,
    min_event_duration: float = STEP_4_ALT_MIN_EVENT_DURATION,
    max_event_duration: float = STEP_4_ALT_MAX_EVENT_DURATION,
    seed: int = STEP_4_SEED,
) -> dict[str, Any]:
    """Generate timeline metadata from millisecond inputs and emit second timestamps."""
    return _generate_timeline_metadata(
        speakers,
        duration,
        overlap_ratios,
        frame_resolution,
        min_event_duration,
        max_event_duration,
        random.Random(seed),
    )


def merge_speaker_groups(groups: Sequence[SpeakerGroup]) -> list[SpeakerGroup]:
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


def fill_speaker_region(
    queue: list[ActiveSegment],
    speaker: str,
    start: float,
    end: float,
    sources: list[dict[str, Any]],
    last_sources: dict[str, dict[str, Any]],
    rng: random.Random,
) -> None:
    gap = STEP_4_SAME_SPEAKER_GAP / 1000.0
    cursor = start
    gain_db = rng.uniform(*STEP_4_GAIN_DB_RANGE)
    while cursor < end - 1e-9:
        segment = queue.pop()
        previous = last_sources.get(speaker)
        continues_previous = (
            previous is not None
            and previous["source_id"] == segment.source_id
            and math.isclose(float(previous["orig_end"]), segment.orig_start, abs_tol=1e-6)
            and math.isclose(float(previous["mix_end"]), cursor, abs_tol=1e-6)
        )
        if previous is not None and not continues_previous:
            cursor = max(cursor, float(previous["mix_end"]) + gap)
            if cursor >= end - 1e-9:
                queue.append(segment)
                break

        duration = min(segment.duration, end - cursor)
        piece = replace(segment, speaker=speaker, orig_end=segment.orig_start + duration)
        source_end = cursor + duration
        if continues_previous:
            previous["orig_end"] = round(piece.orig_end, 6)
            previous["orig_duration"] = round(float(previous["orig_end"]) - float(previous["orig_start"]), 6)
            previous["mix_end"] = round(source_end, 6)
            previous["mix_duration"] = round(float(previous["mix_end"]) - float(previous["mix_start"]), 6)
        else:
            previous = build_source(piece, speaker, len(sources) + 1, cursor, source_end, gain_db)
            sources.append(previous)
            last_sources[speaker] = previous
        if duration < segment.duration - 1e-9:
            queue.append(replace(segment, speaker=speaker, orig_start=segment.orig_start + duration))
        cursor = source_end


def allocate_sources(
    regions: Sequence[Mapping[str, Any]], groups: Sequence[SpeakerGroup], rng: random.Random
) -> list[dict[str, Any]]:
    """Allocate utterance slices once per continuous speaker activity region."""
    group_by_speaker = {group.key: group for group in groups}
    required_speakers = {str(region["speaker"]) for region in regions}

    queues: dict[str, list[ActiveSegment]] = {}
    for speaker in sorted(required_speakers):
        group = group_by_speaker[speaker]
        queues[speaker] = list(group.segments)
        rng.shuffle(queues[speaker])

    sources: list[dict[str, Any]] = []
    last_sources: dict[str, dict[str, Any]] = {}
    for region in regions:
        speaker = str(region["speaker"])
        fill_speaker_region(
            queues[speaker],
            speaker,
            float(region["start"]),
            float(region["end"]),
            sources,
            last_sources,
            rng,
        )
    return sources


def assign_source_groups(
    timeline: dict[str, Any],
    groups: Sequence[SpeakerGroup],
    rng: random.Random,
) -> list[SpeakerGroup]:
    """Select source groups and replace abstract timeline speakers in place."""
    frame_resolution = float(timeline["frame_resolution"])
    required_frames = {
        str(speaker): int(frames)
        for speaker, frames in timeline["speaker_active_frames"].items()
    }
    available = list(groups)
    selected: list[SpeakerGroup] = []
    speaker_mapping: dict[str, str] = {}

    for abstract_speaker, frames in sorted(required_frames.items(), key=lambda item: (-item[1], item[0])):
        required_duration = frames * frame_resolution
        candidates = [
            group
            for group in available
            if group.total_active_duration + 1e-9 >= required_duration
        ]
        if not candidates:
            longest_available = max(
                (group.total_active_duration for group in available),
                default=0.0,
            )
            raise RuntimeError(
                f"Cannot allocate {len(required_frames)} distinct speaker groups: "
                f"one speaker needs {required_duration:.2f} s of active speech, "
                f"but the longest unused group has {longest_available:.2f} s."
            )
        group = weighted_choice_without_replacement(candidates, 1, rng)[0]
        available.remove(group)
        selected.append(group)
        speaker_mapping[abstract_speaker] = group.key

    timeline["speakers"] = [speaker_mapping[str(speaker)] for speaker in timeline["speakers"]]
    for event in timeline["events"]:
        event["speakers"] = [speaker_mapping[str(speaker)] for speaker in event["speakers"]]
    timeline.pop("speaker_active_frames")
    validate_timeline_metadata(timeline)
    return selected


def build_candidate_mixture(
    uri: str,
    groups: Sequence[SpeakerGroup],
    rng: random.Random,
    count_ratios: Sequence[float],
    min_mixture_duration: float,
    max_mixture_duration: float,
    max_speakers: int,
    max_speakers_per_frame: int,
    frame_resolution: float = STEP_4_ALT_FRAME_RESOLUTION,
    min_event_duration: float = STEP_4_ALT_MIN_EVENT_DURATION,
    max_event_duration: float = STEP_4_ALT_MAX_EVENT_DURATION,
    min_overlap_order_duration: float = STEP_4_ALT_MIN_OVERLAP_ORDER_DURATION,
    min_duration_ratio_threshold: float = STEP_4_ALT_MIN_DURATION_RATIO_THRESHOLD,
    max_session_duration: float = STEP_4_ALT_MAX_SESSION_DURATION,
) -> dict[str, Any]:
    if max_speakers < max_speakers_per_frame:
        raise ValueError("max_speakers must be at least max_speakers_per_frame.")
    active_ratios = normalize_count_ratios(count_ratios, max_speakers_per_frame)
    frame_ratios = {
        order: active_ratios[order - 1]
        for order in range(1, max_speakers_per_frame + 1)
    }

    configured_min_duration = min_mixture_duration
    configured_max_duration = max_mixture_duration
    required_duration, effective_min_duration, effective_max_duration = calculate_session_duration_range(
        frame_ratios,
        configured_min_duration,
        configured_max_duration,
        min_overlap_order_duration,
        min_duration_ratio_threshold,
        max_session_duration,
        frame_resolution,
    )
    duration = rng.randint(
        round(effective_min_duration / frame_resolution),
        round(effective_max_duration / frame_resolution),
    ) * frame_resolution

    abstract_speakers = [f"speaker_{index:04d}" for index in range(max_speakers)]
    timeline = _generate_timeline_metadata(
        abstract_speakers,
        duration,
        frame_ratios,
        frame_resolution,
        min_event_duration,
        max_event_duration,
        rng,
    )
    selected = assign_source_groups(timeline, groups, rng)
    speaker_regions = build_speaker_activity_regions(timeline["events"], frame_resolution)
    validate_speaker_activity_regions(timeline["events"], speaker_regions)
    sources = allocate_sources(speaker_regions, selected, rng)
    constrained_orders = [
        order for order, ratio in frame_ratios.items()
        if order > 0 and ratio > 0.0 and ratio >= min_duration_ratio_threshold
    ]
    rare_orders = [
        order for order, ratio in frame_ratios.items()
        if order > 0 and 0.0 < ratio < min_duration_ratio_threshold
    ]

    return {
        "uri": uri,
        "sample_rate": STEP_1_AUDIO_SAMPLE_RATE,
        **timeline,
        "speaker_regions": speaker_regions,
        "same_speaker_gap": STEP_4_SAME_SPEAKER_GAP / 1000.0,
        "duration_policy": {
            "configured_min_duration": configured_min_duration / 1000.0,
            "configured_max_duration": configured_max_duration / 1000.0,
            "minimum_overlap_order_duration": min_overlap_order_duration / 1000.0,
            "minimum_duration_ratio_threshold": min_duration_ratio_threshold,
            "required_duration": required_duration / 1000.0,
            "effective_min_duration": effective_min_duration / 1000.0,
            "effective_max_duration": effective_max_duration / 1000.0,
            "max_session_duration": max_session_duration / 1000.0,
            "duration_constrained_orders": constrained_orders,
            "rare_overlap_orders": rare_orders,
        },
        "max_speakers_per_frame": max_speakers_per_frame,
        "speaker_count": len(timeline["speakers"]),
        "source_count": len(sources),
        "sources": sources,
    }


def resolve_generation_ratios(args: argparse.Namespace) -> list[float]:
    return resolve_count_ratios(
        args.max_speakers_per_frame_ratio,
        args.max_speakers_per_frame,
    )


def summarize_dataset_overlap(mixtures: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, float | int]]:
    """Aggregate exact overlap-order coverage across generated mixtures."""
    frames: Counter[int] = Counter()
    duration_by_order: Counter[int] = Counter()
    total_frames = 0
    for mixture in mixtures:
        resolution = float(mixture["frame_resolution"])
        for order, count in mixture["actual_frames"].items():
            order = int(order)
            count = int(count)
            frames[order] += count
            duration_by_order[order] += count * resolution
            total_frames += count
    return {
        order: {
            "frames": frames[order],
            "duration": round(duration_by_order[order], 9),
            "ratio": frames[order] / total_frames,
        }
        for order in sorted(frames)
    }


def run_metadata_dir(args: argparse.Namespace) -> None:
    groups = merge_speaker_groups(collect_groups(args.vad_dir, args.min_segment_duration))
    max_count = args.max_speakers_per_frame
    active_ratios = resolve_generation_ratios(args)
    rng = random.Random(args.seed)
    mixtures: list[dict[str, Any]] = []

    print(f"   • Speaker groups: {len(groups)}")

    for index in range(args.n_mixtures):
        mixtures.append(
            build_candidate_mixture(
                uri=f"{args.prefix}_{index:08d}",
                groups=groups,
                rng=rng,
                count_ratios=active_ratios,
                min_mixture_duration=args.min_mixture_duration,
                max_mixture_duration=args.max_mixture_duration,
                max_speakers=args.max_speakers,
                max_speakers_per_frame=max_count,
                frame_resolution=args.frame_resolution,
                min_event_duration=args.min_event_duration,
                max_event_duration=args.max_event_duration,
                min_overlap_order_duration=args.min_overlap_order_duration,
                min_duration_ratio_threshold=args.min_duration_ratio_threshold,
                max_session_duration=args.max_session_duration,
            )
        )
        write_progress(index + 1, args.n_mixtures)

    print()
    write_json(mixtures, args.out, force=args.force)
    print("   • Dataset overlap coverage:")
    for order, values in summarize_dataset_overlap(mixtures).items():
        print(
            f"     {order} speakers: {values['frames']} frames, "
            f"{values['duration']:.2f} s, {values['ratio']:.2%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate N-speaker mixtures from exact frame-level overlap quotas.")
    parser.add_argument("--input-dir", dest="vad_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-mixtures", type=int, required=True)
    parser.add_argument("--min-segment-duration", type=float, default=STEP_4_MIN_SEGMENT_DURATION)
    parser.add_argument("--min-mixture-duration", type=float, default=STEP_4_MIN_MIXTURE_DURATION)
    parser.add_argument("--max-mixture-duration", type=float, default=STEP_4_MAX_MIXTURE_DURATION)
    parser.add_argument("--frame-resolution", type=float, default=STEP_4_ALT_FRAME_RESOLUTION)
    parser.add_argument("--min-event-duration", type=float, default=STEP_4_ALT_MIN_EVENT_DURATION)
    parser.add_argument("--max-event-duration", type=float, default=STEP_4_ALT_MAX_EVENT_DURATION)
    parser.add_argument("--seed", type=int, default=STEP_4_SEED)
    parser.add_argument("--prefix", type=str, default="mix")
    parser.add_argument("--max-speakers", type=int, required=True)
    parser.add_argument("--max-speakers-per-frame", type=int, required=True)
    parser.add_argument("--max-speakers-per-frame-ratio", nargs="+")
    parser.add_argument("--min-overlap-order-duration", type=float, default=STEP_4_ALT_MIN_OVERLAP_ORDER_DURATION)
    parser.add_argument("--min-duration-ratio-threshold", type=float, default=STEP_4_ALT_MIN_DURATION_RATIO_THRESHOLD)
    parser.add_argument("--max-session-duration", type=float, default=STEP_4_ALT_MAX_SESSION_DURATION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_ratios = resolve_generation_ratios(args)
    frame_ratio_map = {
        order: active_ratios[order - 1]
        for order in range(1, args.max_speakers_per_frame + 1)
    }
    configured_min_duration = args.min_mixture_duration
    configured_max_duration = args.max_mixture_duration
    _, effective_min_duration, effective_max_duration = calculate_session_duration_range(
        frame_ratio_map,
        configured_min_duration,
        configured_max_duration,
        args.min_overlap_order_duration,
        args.min_duration_ratio_threshold,
        args.max_session_duration,
        args.frame_resolution,
    )

    print(f"🚀 Generate frame-balanced metadata in '{args.vad_dir}'")
    print(f"   • Target mixtures: {args.n_mixtures}")
    print(f"   • Frame ratios: {', '.join(f'{count}: {ratio:.2%}' for count, ratio in frame_ratio_map.items())}")
    print(f"   • Minimum overlap-order coverage: {args.min_overlap_order_duration:g} ms")
    print(f"   • Duration ratio threshold: {args.min_duration_ratio_threshold:.2%}")
    print(f"   • Maximum session duration: {args.max_session_duration:g} ms")
    print(f"   • Same-speaker gap: {STEP_4_SAME_SPEAKER_GAP} ms")
    print(f"   • Configured duration: {configured_min_duration:g}–{configured_max_duration:g} ms")
    print(f"   • Effective duration: {effective_min_duration:g}–{effective_max_duration:g} ms")
    run_metadata_dir(args)
    print(f"   • Output: {args.out}")
    print("✅ Frame-balanced metadata completed.")


if __name__ == "__main__":
    main()

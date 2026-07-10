#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- METADATA ---

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import (
    STEP_1_AUDIO_SAMPLE_RATE,
    STEP_4_GAIN_DB_RANGE,
    STEP_4_MAX_BUILD_TRIALS,
    STEP_4_MAX_MIXTURE_DURATION,
    STEP_4_MAX_SPEAKERS,
    STEP_4_MAX_SPEAKERS_PER_FRAME,
    STEP_4_MIN_MIXTURE_DURATION,
    STEP_4_MIN_SEGMENT_DURATION,
    STEP_4_OVERLAP_RATIO,
    STEP_4_POST_SILENCE,
    STEP_4_PRE_SILENCE,
    STEP_4_RATIO_TRIES,
    STEP_4_OVERLAP_RANDOM_OFFSET,
    STEP_4_SAME_SPEAKER_GAP,
    STEP_4_SEED,
)
from pipeline.utils.progress import write_progress


@dataclass(frozen=True)
class ActiveSegment:
    source_id: str
    audio: str
    audio_derivative: str
    speaker: str | None
    segment_index: int
    orig_start: float
    orig_end: float

    @property
    def duration(self) -> float:
        return self.orig_end - self.orig_start


@dataclass(frozen=True)
class SpeakerGroup:
    key: str
    segments: list[ActiveSegment]
    total_active_duration: float


def format_usage_histogram(usage_counts: Counter[str], max_items: int = 12, bar_width: int = 24) -> list[str]:
    if not usage_counts:
        return ["   • Source usage counts: (none)"]

    items = sorted(usage_counts.items(), key=lambda item: (-item[1], item[0]))[:max_items]
    highest = max(count for _, count in items) or 1
    lines = ["   • Source usage counts:"]
    for name, count in items:
        filled = max(1, int(round(bar_width * count / highest)))
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"     {name:<24} |{bar}| {count}")
    remaining = len(usage_counts) - len(items)
    if remaining > 0:
        lines.append(f"     ... ({remaining} more)")
    return lines


def load_vad_json(path: Path) -> list[ActiveSegment]:
    """
    Load one VAD JSON file as active segments.

    Args:
        path: Input VAD JSON path.

    Returns:
        List of active speech segments.
    """
    with path.open("r", encoding="utf-8") as f:
        item = json.load(f)

    sr = int(item.get("sampling_rate", STEP_1_AUDIO_SAMPLE_RATE))
    if sr != STEP_1_AUDIO_SAMPLE_RATE:
        raise ValueError(f"Unexpected sampling_rate={sr} in {path}; expected {STEP_1_AUDIO_SAMPLE_RATE}.")

    audio = str(item["input"])
    audio_derivative = Path(audio).stem
    speaker = item["speaker"]
    speaker = str(speaker) if speaker not in (None, "") else None

    segments: list[ActiveSegment] = []
    for i, seg in enumerate(item.get("segments", [])):
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except Exception:
            continue
        if end > start:
            segments.append(
                ActiveSegment(
                    source_id=f"{audio_derivative}_seg{i:04d}",
                    audio=audio,
                    audio_derivative=audio_derivative,
                    speaker=speaker,
                    segment_index=i,
                    orig_start=start,
                    orig_end=end,
                )
            )
    return segments


def collect_groups(vad_root: Path, min_duration_ms: float) -> list[SpeakerGroup]:
    """
    Group VAD segments by source audio.

    Args:
        vad_root: Directory containing VAD JSON files.
        min_duration_ms: Minimum segment duration in milliseconds.

    Returns:
        Speaker groups sorted by audio derivative.
    """
    min_duration_sec = min_duration_ms / 1000.0
    paths = sorted(vad_root.rglob("*.json"))
    grouped: dict[str, list[ActiveSegment]] = {}
    for path in paths:
        if not path.name.endswith("_std_nml_vad.json"):
            continue
        for seg in load_vad_json(path):
            if seg.duration >= min_duration_sec:
                grouped.setdefault(seg.audio_derivative, []).append(seg)

    groups: list[SpeakerGroup] = []
    for key in sorted(grouped):
        segments = sorted(grouped[key], key=lambda seg: (seg.orig_start, seg.segment_index))
        groups.append(
            SpeakerGroup(
                key=key,
                segments=segments,
                total_active_duration=sum(seg.duration for seg in segments),
            )
        )
    return groups


def sample_overlap_ratio(rng: random.Random, center: float, offset: float) -> float:
    """
    Sample one overlap ratio around a center value.

    Args:
        rng: Random number generator.
        center: Center overlap ratio.
        offset: Random offset around the center.

    Returns:
        A sampled overlap ratio.
    """
    return rng.uniform(max(0.0, min(1.0, center - offset)), max(0.0, min(1.0, center + offset)))


def weighted_choice_without_replacement(
    items: list[SpeakerGroup],
    count: int,
    rng: random.Random,
) -> list[SpeakerGroup]:
    """
    Pick speaker groups without replacement using duration as weight.

    Args:
        items: Candidate speaker groups.
        count: Maximum number of groups to select.
        rng: Random number generator.

    Returns:
        Selected speaker groups.
    """
    pool = list(items)
    chosen: list[SpeakerGroup] = []
    for _ in range(min(count, len(pool))):
        weights = [max(item.total_active_duration, 0.0) for item in pool]
        total = sum(weights)
        if total <= 0:
            chosen.append(pool.pop(rng.randrange(len(pool))))
            continue

        target = rng.random() * total
        acc = 0.0
        index = len(pool) - 1
        for i, weight in enumerate(weights):
            acc += weight
            if acc >= target:
                index = i
                break
        chosen.append(pool.pop(index))
    return chosen


def build_source(
    seg: ActiveSegment,
    speaker: str,
    idx: int,
    start: float,
    end: float,
    gain_db: float,
) -> dict[str, Any]:
    """
    Convert one segment into the metadata source format.

    Args:
        seg: Active segment to serialize.
        speaker: Speaker identifier for the mixture.
        idx: 1-based source index.
        start: Mix start time in seconds.
        end: Mix end time in seconds.
        gain_db: Gain applied to the source.

    Returns:
        Source metadata dictionary.
    """
    return {
        "source": f"s{idx}",
        "source_id": seg.source_id,
        "audio": seg.audio,
        "audio_derivative": seg.audio_derivative,
        "speaker": speaker,
        "segment_index": seg.segment_index,
        "orig_start": round(seg.orig_start, 6),
        "orig_end": round(seg.orig_end, 6),
        "orig_duration": round(seg.duration, 6),
        "mix_start": round(start, 6),
        "mix_end": round(end, 6),
        "mix_duration": round(seg.duration, 6),
        "gain_db": round(gain_db, 3),
    }


def find_fit(
    queue: list[ActiveSegment],
    prev: dict[str, Any] | None,
    same_speaker: bool,
    overlap_center: float,
    overlap_offset: float,
    max_duration: float,
    rng: random.Random,
) -> tuple[int, float, float, float | None] | None:
    """
    Find a segment placement that fits the current mixture.

    Args:
        queue: Remaining segments for one speaker group.
        prev: Previously placed source, if any.
        same_speaker: Whether the next segment belongs to the same speaker.
        overlap_center: Center overlap ratio.
        overlap_offset: Random overlap offset.
        max_duration: Mixture duration limit in seconds.
        rng: Random number generator.

    Returns:
        A tuple of (queue index, start, end, ratio) or None if no fit is found.
    """
    if same_speaker:
        start = prev["mix_end"] + (STEP_4_SAME_SPEAKER_GAP / 1000.0)
        for i, seg in enumerate(queue):
            end = start + seg.duration
            if end <= max_duration:
                return i, start, end, None
        return None

    prev_end = float(prev["mix_end"]) if prev is not None else STEP_4_PRE_SILENCE / 1000.0
    prev_duration = float(prev["orig_duration"]) if prev is not None else 0.0
    for i, seg in enumerate(queue):
        for _ in range(STEP_4_RATIO_TRIES):
            ratio = sample_overlap_ratio(rng, overlap_center, overlap_offset)
            overlap = ratio * min(prev_duration, seg.duration)
            start = prev_end - overlap
            end = start + seg.duration
            if end <= max_duration:
                return i, start, end, ratio
    return None


def compute_global_overlap(sources: list[dict[str, Any]], max_speakers_per_frame: int) -> tuple[list[dict[str, Any]], bool]:
    """
    Compute overlap intervals and verify the per-frame speaker limit.

    Args:
        sources: Mixture sources.
        max_speakers_per_frame: Maximum number of overlapping speakers allowed.

    Returns:
        A tuple of (overlap intervals, ok flag).
    """
    events: list[tuple[float, int, str, str]] = []
    for src in sources:
        events.append((float(src["mix_start"]), 1, str(src["speaker"]), str(src["source"])))
        events.append((float(src["mix_end"]), 0, str(src["speaker"]), str(src["source"])))

    events.sort(key=lambda item: (item[0], item[1]))
    active = Counter()
    intervals: list[dict[str, Any]] = []
    prev_time: float | None = None

    for time, kind, speaker, source_id in events:
        if prev_time is not None and time > prev_time:
            count = sum(active.values())
            if count > max_speakers_per_frame:
                return intervals, False
            if count >= 2:
                intervals.append(
                    {
                        "start": round(prev_time, 6),
                        "end": round(time, 6),
                        "duration": round(time - prev_time, 6),
                        "count": count,
                        "sources": sorted(active.elements()),
                    }
                )

        if kind == 0:
            if active[source_id] <= 1:
                active.pop(source_id, None)
            else:
                active[source_id] -= 1
        else:
            active[source_id] += 1
        prev_time = time

    return intervals, True


def build_candidate_mixture(
    input_dir: Path,
    uri: str,
    groups: list[SpeakerGroup],
    rng: random.Random,
    run_uuid: str,
    overlap_center: float = STEP_4_OVERLAP_RATIO,
    overlap_offset: float = STEP_4_OVERLAP_RANDOM_OFFSET,
    min_mixture_duration: float = STEP_4_MIN_MIXTURE_DURATION,
    max_mixture_duration: float = STEP_4_MAX_MIXTURE_DURATION,
    max_speakers: int = STEP_4_MAX_SPEAKERS,
    max_speakers_per_frame: int = STEP_4_MAX_SPEAKERS_PER_FRAME,
) -> dict[str, Any] | None:
    """
    Build one valid mixture metadata record.

    Args:
        uri: Mixture identifier.
        groups: Candidate speaker groups.
        rng: Random number generator.
        overlap_center: Center overlap ratio.
        overlap_offset: Random overlap offset.
        min_mixture_duration: Minimum mixture duration in milliseconds.
        max_mixture_duration: Maximum mixture duration in milliseconds.
        max_speakers: Maximum number of speakers in one mixture.
        max_speakers_per_frame: Maximum number of overlapping speakers in one frame.

    Returns:
        Mixture metadata dictionary, or None when no valid candidate is found.
    """
    if len(groups) < 2:
        return None

    min_mix_sec = min_mixture_duration / 1000.0
    max_mix_sec = max_mixture_duration / 1000.0
    ordered = weighted_choice_without_replacement(groups, 2, rng)

    def add_group() -> bool:
        remaining = [group for group in groups if group.key not in {item.key for item in ordered}]
        if not remaining:
            return False
        ordered.append(weighted_choice_without_replacement(remaining, 1, rng)[0])
        return True

    while True:
        if sum(group.total_active_duration for group in ordered) >= min_mix_sec and len(ordered) >= 2:
            break
        if len(ordered) >= min(max_speakers, len(groups)) or not add_group():
            break

    queues = {group.key: list(group.segments) for group in ordered}
    for queue in queues.values():
        rng.shuffle(queue)

    keys = [group.key for group in ordered]
    speaker_keys = list(keys)
    rng.shuffle(speaker_keys)
    fallback_ids = {key: f"{uri}_spk{i + 1}_{run_uuid}" for i, key in enumerate(speaker_keys)}

    resolved_speakers: dict[str, str] = {}
    for key, group in zip(keys, ordered, strict=False):
        speaker = next((seg.speaker for seg in group.segments if seg.speaker not in (None, "")), None)
        resolved_speakers[key] = str(speaker) if speaker not in (None, "") else fallback_ids[key]

    sources: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    used: set[str] = set()
    blocked: set[str] = set()
    current_end = STEP_4_PRE_SILENCE / 1000.0
    prev_source: dict[str, Any] | None = None
    prev_key: str | None = None

    while True:
        if len(used) >= 2 and current_end + (STEP_4_POST_SILENCE / 1000.0) >= min_mix_sec:
            break

        available = [key for key in keys if queues.get(key) and key not in blocked]
        if not available:
            break

        candidates = [key for key in available if key != prev_key] or available
        key = rng.choice(candidates)
        if prev_key is not None and key == prev_key and len(available) > 1:
            blocked.add(key)
            continue

        fit = find_fit(
            queue=queues[key],
            prev=prev_source,
            same_speaker=prev_key is not None and key == prev_key,
            overlap_center=overlap_center,
            overlap_offset=overlap_offset,
            max_duration=max_mix_sec,
            rng=rng,
        )
        if fit is None:
            blocked.add(key)
            continue

        idx, start, end, ratio = fit
        seg = queues[key].pop(idx)
        source = build_source(
            seg,
            resolved_speakers[key],
            len(sources) + 1,
            start,
            end,
            rng.uniform(*STEP_4_GAIN_DB_RANGE),
        )
        sources.append(source)
        used.add(key)

        if prev_source is not None:
            overlap_start = max(float(prev_source["mix_start"]), float(source["mix_start"]))
            overlap_end = min(float(prev_source["mix_end"]), float(source["mix_end"]))
            overlap_duration = max(0.0, overlap_end - overlap_start)
            denom = min(float(prev_source["orig_duration"]), float(source["orig_duration"]))
            overlaps.append(
                {
                    "between": [prev_source["source"], source["source"]],
                    "start": round(overlap_start, 6),
                    "end": round(overlap_end, 6),
                    "duration": round(overlap_duration, 6),
                    "ratio_target": ratio,
                    "ratio_actual": round(overlap_duration / denom, 6) if denom > 0 else None,
                }
            )

        prev_source = source
        prev_key = key
        current_end = float(source["mix_end"])

    if len(sources) < 2 or len(used) < 2:
        return None

    mix_end = max(float(src["mix_end"]) for src in sources)
    duration = round(mix_end + (STEP_4_POST_SILENCE / 1000.0), 6)
    if mix_end > max_mix_sec or not (min_mix_sec <= duration <= max_mix_sec):
        return None

    global_overlap, ok = compute_global_overlap(sources, max_speakers_per_frame)
    if not ok:
        return None

    speakers = list(dict.fromkeys(resolved_speakers[key] for key in keys if key in used))

    return {
        "uri": uri,
        "uuid": run_uuid,
        "input": str(input_dir),
        "duration": duration,
        "sample_rate": STEP_1_AUDIO_SAMPLE_RATE,
        "overlap_ratio_center": overlap_center,
        "overlap_ratio_offset": overlap_offset,
        "pre_silence": round(STEP_4_PRE_SILENCE, 6),
        "post_silence": round(STEP_4_POST_SILENCE, 6),
        "max_speakers": STEP_4_MAX_SPEAKERS,
        "max_speakers_per_frame": STEP_4_MAX_SPEAKERS_PER_FRAME,
        "speakers": speakers,
        "speaker_count": len(speakers),
        "source_count": len(sources),
        "overlap_count": len(overlaps),
        "global_overlap_count": len(global_overlap),
        "sources": sources,
        "overlap": overlaps,
        "global_overlap": global_overlap,
    }


def write_json(items: list[dict[str, Any]], path: Path, force: bool = False) -> None:
    """
    Write the generated metadata list to disk.

    Args:
        items: Mixture metadata records.
        path: Output JSON path.
        force: Overwrite existing file when True.
    """
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate metadata generation arguments.

    Args:
        args: Parsed CLI arguments.
    """
    if args.min_mixture_duration <= 0:
        raise ValueError("--min-mixture-duration must be positive.")
    if args.max_mixture_duration < args.min_mixture_duration:
        raise ValueError("--max-mixture-duration must be >= --min-mixture-duration.")
    if args.min_segment_duration < 0:
        raise ValueError("--min-segment-duration must be non-negative.")
    if args.overlap_random_offset < 0:
        raise ValueError("--overlap-random-offset must be non-negative.")
    if args.max_speakers < 2:
        raise ValueError("--max-speakers must be at least 2.")
    if args.max_speakers_per_frame < 1:
        raise ValueError("--max-speakers-per-frame must be at least 1.")


def run_metadata_dir(args: argparse.Namespace) -> None:
    """
    Step 4: Generate mixture metadata from VAD JSON files.

    Args:
        args: Parsed CLI arguments.
    """
    groups = collect_groups(args.vad_dir, args.min_segment_duration)
    if len(groups) < 2:
        raise RuntimeError("Need at least two speaker groups.")

    rng = random.Random(args.seed)
    run_uuid = uuid.uuid4().hex[:8]
    mixtures: list[dict[str, Any]] = []
    attempts = 0

    print(f"🚀 Generate metadata in '{args.vad_dir}'")
    print(f"   • Speaker groups: {len(groups)}")
    print(f"   • Target mixtures: {args.n_mixtures}")

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
            overlap_center=args.overlap_ratio,
            overlap_offset=args.overlap_random_offset,
            min_mixture_duration=args.min_mixture_duration,
            max_mixture_duration=args.max_mixture_duration,
            max_speakers=args.max_speakers,
            max_speakers_per_frame=args.max_speakers_per_frame,
        )
        if record is not None:
            mixtures.append(record)
            write_progress(len(mixtures), args.n_mixtures)

    write_json(mixtures, args.out, force=args.force)

    usage_counts = Counter(
        name for mixture in mixtures for name in {src["audio_derivative"] for src in mixture["sources"]}
    )
    print()
    print(f"   • Active segments: {sum(len(group.segments) for group in groups)}")
    print(f"   • Attempts: {attempts}")
    for line in format_usage_histogram(usage_counts):
        print(line)
    print(f"   • Output: {args.out}")
    print("✅ Metadata completed.")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VAD-based mixture metadata.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to _std_nml_vad.json files produced by step_3_vad.py.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--n-mixtures", type=int, required=True, help="Number of mixtures to generate.")
    parser.add_argument("--overlap-ratio", type=float, default=STEP_4_OVERLAP_RATIO, help="Center overlap ratio.")
    parser.add_argument("--overlap-random-offset", type=float, default=STEP_4_OVERLAP_RANDOM_OFFSET, help="Random offset around overlap ratio.")
    parser.add_argument("--min-segment-duration", dest="min_segment_duration", type=float, default=STEP_4_MIN_SEGMENT_DURATION, help="Minimum active segment duration (ms).")
    parser.add_argument("--min-mixture-duration", type=float, default=STEP_4_MIN_MIXTURE_DURATION, help="Minimum mixture duration (ms).")
    parser.add_argument("--max-mixture-duration", type=float, default=STEP_4_MAX_MIXTURE_DURATION, help="Maximum mixture duration (ms).")
    parser.add_argument("--seed", type=int, default=STEP_4_SEED, help="Random seed.")
    parser.add_argument("--prefix", type=str, default="mix", help="Mixture URI prefix.")
    parser.add_argument("--max-build-trials", type=int, default=STEP_4_MAX_BUILD_TRIALS, help="Maximum sampling attempts.")
    parser.add_argument("--max-speakers", dest="max_speakers", type=int, default=STEP_4_MAX_SPEAKERS, help="Maximum speakers per mixture.")
    parser.add_argument("--max-speakers-per-frame", dest="max_speakers_per_frame", type=int, default=STEP_4_MAX_SPEAKERS_PER_FRAME, help="Maximum overlapping speakers per frame.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_metadata_dir(args)


if __name__ == "__main__":
    main()

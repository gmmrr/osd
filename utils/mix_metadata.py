#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from config.params import AUDIO_SAMPLE_RATE, MIX_POST_SILENCE, MIX_PRE_SILENCE, MIX_SAME_SPEAKER_GAP
except ModuleNotFoundError:  # pragma: no cover - convenience for `uv run path/to/script.py`
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config.params import AUDIO_SAMPLE_RATE, MIX_POST_SILENCE, MIX_PRE_SILENCE, MIX_SAME_SPEAKER_GAP


DEFAULT_MAX_SPEAKERS = 3
DEFAULT_MAX_SPEAKERS_PER_FRAME = 2
DEFAULT_RATIO_TRIES = 8


@dataclass(frozen=True)
class ActiveSegment:
    source_id: str
    audio: str
    audio_derivative: str
    speaker_id: str | None
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


def _match_audio_ref(audio_path: Path, value: Any) -> bool:
    if value in (None, ""):
        return False

    text = str(value).strip()
    if not text:
        return False

    stem = audio_path.stem
    base = stem.removesuffix("_nml").removesuffix("_std")
    name = audio_path.name
    resolved = str(audio_path.resolve())
    path_value = Path(text)

    return any(
        [
            text == name,
            text == stem,
            text == base,
            path_value.name == name,
            path_value.stem in {stem, base},
            resolved == text,
            text.endswith(name),
            text.endswith(stem),
            text.endswith(base),
        ]
    )


def infer_speaker_id(audio_path: Path) -> str | None:
    for parent in [audio_path.parent, *audio_path.parents]:
        for meta_path in (parent / "metadata.jsonl", parent / "metadata.csv"):
            if not meta_path.exists():
                continue

            try:
                if meta_path.suffix == ".jsonl":
                    with meta_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            if isinstance(row, dict) and (
                                _match_audio_ref(audio_path, row.get("wav_path"))
                                or _match_audio_ref(audio_path, row.get("input"))
                            ):
                                speaker_id = row.get("speaker_id")
                                return str(speaker_id) if speaker_id not in (None, "") else None
                else:
                    with meta_path.open("r", encoding="utf-8", newline="") as f:
                        for row in csv.DictReader(f):
                            if _match_audio_ref(audio_path, row.get("wav_path")) or _match_audio_ref(
                                audio_path, row.get("input")
                            ):
                                speaker_id = row.get("speaker_id")
                                return str(speaker_id) if speaker_id not in (None, "") else None
            except Exception:
                continue

    return None


def load_vad_json(path: Path, require_speaker_id: bool = False) -> list[ActiveSegment]:
    with path.open("r", encoding="utf-8") as f:
        item = json.load(f)

    sr = int(item.get("sampling_rate", AUDIO_SAMPLE_RATE))
    if sr != AUDIO_SAMPLE_RATE:
        raise ValueError(f"Unexpected sampling_rate={sr} in {path}; expected {AUDIO_SAMPLE_RATE}.")

    audio = str(item["input"])
    audio_derivative = str(item.get("audio_derivative", Path(audio).stem))

    raw_speaker_id = item.get("speaker_id")
    if raw_speaker_id in (None, "") and isinstance(item.get("parameters"), dict):
        raw_speaker_id = item["parameters"].get("speaker_id")
    raw_speaker_id = str(raw_speaker_id) if raw_speaker_id not in (None, "") else None

    if require_speaker_id and (raw_speaker_id if raw_speaker_id is not None else infer_speaker_id(Path(audio))) in (
        None,
        "",
    ):
        raise ValueError(f"speaker_id is null for {path}")

    segments: list[ActiveSegment] = []
    for i, seg in enumerate(item.get("segments", [])):
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except Exception:
            continue
        if end <= start:
            continue

        segments.append(
            ActiveSegment(
                source_id=f"{audio_derivative}_seg{i:04d}",
                audio=audio,
                audio_derivative=audio_derivative,
                speaker_id=raw_speaker_id,
                segment_index=i,
                orig_start=start,
                orig_end=end,
            )
        )

    return segments


def collect_groups(
    vad_root: Path,
    vad_pattern: str,
    min_duration: float,
    require_speaker_id: bool = False,
) -> list[SpeakerGroup]:
    grouped: dict[str, list[ActiveSegment]] = {}
    for path in sorted(vad_root.rglob(vad_pattern)):
        for seg in load_vad_json(path, require_speaker_id=require_speaker_id):
            if seg.duration < min_duration:
                continue
            grouped.setdefault(seg.audio_derivative, []).append(seg)

    return [
        SpeakerGroup(
            key=key,
            segments=ordered,
            total_active_duration=sum(seg.duration for seg in ordered),
        )
        for key, ordered in (
            (key, sorted(grouped[key], key=lambda seg: (seg.orig_start, seg.segment_index)))
            for key in sorted(grouped)
        )
    ]


def clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def sample_overlap_ratio(rng: random.Random, center: float, offset: float) -> float:
    low = clamp_ratio(center - offset)
    high = clamp_ratio(center + offset)
    return rng.uniform(low, high)


def weighted_sample_without_replacement(
    items: list[SpeakerGroup],
    count: int,
    rng: random.Random,
) -> list[SpeakerGroup]:
    pool = list(items)
    chosen: list[SpeakerGroup] = []
    for _ in range(min(count, len(pool))):
        weights = [max(item.total_active_duration, 0.0) for item in pool]
        total = sum(weights)
        if total <= 0:
            index = rng.randrange(len(pool))
        else:
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


def _make_source_record(
    uri: str,
    source_name: str,
    seg: ActiveSegment,
    speaker_label: str,
    mix_start: float,
    mix_end: float,
    gain_db: float,
) -> dict[str, Any]:
    return {
        "source": source_name,
        "source_id": seg.source_id,
        "audio": seg.audio,
        "audio_derivative": seg.audio_derivative,
        "speaker_id": seg.speaker_id,
        "segment_index": seg.segment_index,
        "speaker": speaker_label,
        "orig_start": round(seg.orig_start, 6),
        "orig_end": round(seg.orig_end, 6),
        "orig_duration": round(seg.duration, 6),
        "mix_start": round(mix_start, 6),
        "mix_end": round(mix_end, 6),
        "mix_duration": round(seg.duration, 6),
        "gain_db": round(gain_db, 3),
    }


def _find_fitting_segment(
    queue: list[ActiveSegment],
    prev_source: dict[str, Any] | None,
    current_end: float,
    same_speaker: bool,
    overlap_center: float,
    overlap_offset: float,
    max_duration: float,
    same_speaker_gap: float,
    rng: random.Random,
    ratio_tries: int,
) -> tuple[int, float, float, float | None] | None:
    if same_speaker:
        base_start = prev_source["mix_end"] + same_speaker_gap if prev_source is not None else current_end
        for idx, seg in enumerate(queue):
            mix_end = base_start + seg.duration
            if mix_end <= max_duration:
                return idx, base_start, mix_end, None
        return None

    prev_duration = float(prev_source["orig_duration"]) if prev_source is not None else 0.0
    prev_end = float(prev_source["mix_end"]) if prev_source is not None else current_end

    for idx, seg in enumerate(queue):
        for _ in range(ratio_tries):
            ratio = sample_overlap_ratio(rng, overlap_center, overlap_offset)
            overlap = ratio * min(prev_duration, seg.duration)
            mix_start = prev_end - overlap
            mix_end = mix_start + seg.duration
            if mix_end <= max_duration:
                return idx, mix_start, mix_end, ratio

    return None


def _build_sources_from_groups(
    uri: str,
    groups: list[SpeakerGroup],
    rng: random.Random,
    overlap_center: float,
    overlap_offset: float,
    min_mixture_duration: float,
    max_mixture_duration: float,
    max_speakers_per_frame: int,
    gain_db_range: tuple[float, float] = (-3.0, 3.0),
) -> dict[str, Any] | None:
    if len(groups) < 2:
        return None

    queues: dict[str, list[ActiveSegment]] = {group.key: list(group.segments) for group in groups}
    for queue in queues.values():
        rng.shuffle(queue)

    order = [group.key for group in groups]
    label_order = list(order)
    rng.shuffle(label_order)
    label_map = {key: f"{uri}_spk{i + 1}" for i, key in enumerate(label_order)}

    sources: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    used_speakers: set[str] = set()
    blocked: set[str] = set()

    current_end = MIX_PRE_SILENCE
    prev_source: dict[str, Any] | None = None
    prev_key: str | None = None

    while True:
        final_duration = current_end + MIX_POST_SILENCE
        speaker_count = len(used_speakers)
        if speaker_count >= 2 and final_duration >= min_mixture_duration:
            break

        available = {key for key in order if queues.get(key) and key not in blocked}
        if not available:
            break

        candidates = [key for key in available if key != prev_key]
        next_key = rng.choice(candidates or list(available))
        if next_key is None:
            break

        same_speaker = prev_key is not None and next_key == prev_key
        if same_speaker and len(available) > 1:
            blocked.add(next_key)
            continue

        queue = queues[next_key]
        fit = _find_fitting_segment(
            queue=queue,
            prev_source=prev_source,
            current_end=current_end,
            same_speaker=same_speaker,
            overlap_center=overlap_center,
            overlap_offset=overlap_offset,
            max_duration=max_mixture_duration,
            same_speaker_gap=MIX_SAME_SPEAKER_GAP,
            rng=rng,
            ratio_tries=DEFAULT_RATIO_TRIES,
        )
        if fit is None:
            blocked.add(next_key)
            continue

        idx, mix_start, mix_end, ratio = fit
        seg = queue.pop(idx)
        gain_db = rng.uniform(*gain_db_range)
        source_name = f"s{len(sources) + 1}"
        source = _make_source_record(uri, source_name, seg, label_map[next_key], mix_start, mix_end, gain_db)
        sources.append(source)
        used_speakers.add(next_key)

        if prev_source is not None:
            prev_end = float(prev_source["mix_end"])
            curr_start = float(source["mix_start"])
            curr_end = float(source["mix_end"])
            overlap_start = max(float(prev_source["mix_start"]), curr_start)
            overlap_end = min(prev_end, curr_end)
            overlap_duration = max(0.0, overlap_end - overlap_start)
            prev_duration = float(prev_source["orig_duration"])
            curr_duration = float(source["orig_duration"])
            denom = min(prev_duration, curr_duration)
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
        prev_key = next_key
        current_end = float(source["mix_end"])

    speaker_labels = [label_map[key] for key in order if any(src["speaker"] == label_map[key] for src in sources)]
    speaker_ids = sorted(
        {
            src["speaker_id"]
            for src in sources
            if src["speaker_id"] not in (None, "")
        }
    )

    if len(sources) < 2 or len(speaker_labels) < 2:
        return None

    mix_end = max(float(src["mix_end"]) for src in sources)
    if mix_end > max_mixture_duration:
        return None

    global_overlap, ok = compute_global_overlap(sources, max_speakers_per_frame)
    if not ok:
        return None

    pre_silence = round(MIX_PRE_SILENCE, 6)
    post_silence = round(MIX_POST_SILENCE, 6)
    duration = round(mix_end + post_silence, 6)
    if duration < min_mixture_duration or duration > max_mixture_duration:
        return None

    return {
        "uri": uri,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "duration": duration,
        "overlap_ratio_center": overlap_center,
        "overlap_ratio_offset": overlap_offset,
        "pre_silence": pre_silence,
        "post_silence": post_silence,
        "max_speakers": DEFAULT_MAX_SPEAKERS,
        "max_speakers_per_frame": DEFAULT_MAX_SPEAKERS_PER_FRAME,
        "max_number_speaker": DEFAULT_MAX_SPEAKERS,
        "max_overlap_speaker": DEFAULT_MAX_SPEAKERS_PER_FRAME,
        "speaker_count": len(speaker_labels),
        "speaker_labels": speaker_labels,
        "speaker_ids": speaker_ids,
        "source_count": len(sources),
        "sources": sources,
        "overlap": overlaps,
        "global_overlap": global_overlap,
    }


def compute_global_overlap(
    sources: list[dict[str, Any]],
    max_speakers_per_frame: int,
) -> tuple[list[dict[str, Any]], bool]:
    events: list[tuple[float, int, str, str]] = []
    for src in sources:
        start = float(src["mix_start"])
        end = float(src["mix_end"])
        speaker = str(src["speaker"])
        source_id = str(src["source"])
        events.append((start, 1, speaker, source_id))
        events.append((end, 0, speaker, source_id))

    events.sort(key=lambda item: (item[0], item[1]))

    active = Counter[str]()
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
    uri: str,
    groups: list[SpeakerGroup],
    rng: random.Random,
    overlap_center: float,
    overlap_offset: float,
    min_mixture_duration: float,
    max_mixture_duration: float,
    max_speakers: int,
    max_speakers_per_frame: int,
) -> dict[str, Any] | None:
    if len(groups) < 2:
        return None

    ordered = weighted_sample_without_replacement(groups, 2, rng)

    def add_next_group() -> bool:
        remaining = [group for group in groups if group.key not in {item.key for item in ordered}]
        if not remaining:
            return False
        ordered.append(weighted_sample_without_replacement(remaining, 1, rng)[0])
        return True

    while True:
        total_active = sum(group.total_active_duration for group in ordered)
        need_more_speakers = total_active < min_mixture_duration and len(ordered) < min(max_speakers, len(groups))

        if need_more_speakers:
            if not add_next_group():
                break
            continue

        record = _build_sources_from_groups(
            uri=uri,
            groups=ordered,
            rng=rng,
            overlap_center=overlap_center,
            overlap_offset=overlap_offset,
            min_mixture_duration=min_mixture_duration,
            max_mixture_duration=max_mixture_duration,
            max_speakers_per_frame=max_speakers_per_frame,
        )
        if record is None:
            if len(ordered) >= min(max_speakers, len(groups)):
                return None
            if not add_next_group():
                return None
            continue

        if record["duration"] >= min_mixture_duration:
            return record

        if len(ordered) >= min(max_speakers, len(groups)):
            return None
        if not add_next_group():
            return None

    return None


def write_json(items: list[dict[str, Any]], path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VAD-based mixture metadata.")
    parser.add_argument("--vad-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-mixtures", type=int, default=5)
    parser.add_argument(
        "--overlap-ratios",
        type=float,
        nargs=2,
        metavar=("CENTER", "OFFSET"),
        default=(0.2, 0.0),
        help="Center and random offset for overlap ratio. Example: 0.4 0.1 -> uniform(0.3, 0.5).",
    )
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--min-mixture-duration", type=float, default=15.0)
    parser.add_argument("--max-mixture-duration", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="mix")
    parser.add_argument("--max-build-trials", type=int, default=10000)
    parser.add_argument(
        "--max-speakers",
        "--max-number-speaker",
        dest="max_speakers",
        type=int,
        default=DEFAULT_MAX_SPEAKERS,
    )
    parser.add_argument(
        "--max-speakers-per-frame",
        "--max-overlap-speaker",
        dest="max_speakers_per_frame",
        type=int,
        default=DEFAULT_MAX_SPEAKERS_PER_FRAME,
    )
    parser.add_argument("--require-speaker-id", action="store_true")
    parser.add_argument("--vad-pattern", type=str, default="*_vad.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_mixture_duration <= 0:
        raise ValueError("--min-mixture-duration must be positive.")
    if args.max_mixture_duration < args.min_mixture_duration:
        raise ValueError("--max-mixture-duration must be >= --min-mixture-duration.")
    if args.min_duration < 0:
        raise ValueError("--min-duration must be non-negative.")
    if args.overlap_ratios[1] < 0:
        raise ValueError("--overlap-ratios OFFSET must be non-negative.")
    if args.max_speakers < 2:
        raise ValueError("--max-speakers must be at least 2.")
    if args.max_speakers_per_frame < 1:
        raise ValueError("--max-speakers-per-frame must be at least 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = random.Random(args.seed)
    groups = collect_groups(
        vad_root=args.vad_dir,
        vad_pattern=args.vad_pattern,
        min_duration=args.min_duration,
        require_speaker_id=args.require_speaker_id,
    )
    if len(groups) < 2:
        raise RuntimeError("Need at least two speaker groups.")

    overlap_center, overlap_offset = args.overlap_ratios

    mixtures: list[dict[str, Any]] = []
    attempts = 0
    while len(mixtures) < args.n_mixtures:
        attempts += 1
        if attempts > args.max_build_trials:
            raise RuntimeError(f"Only generated {len(mixtures)} mixtures after {args.max_build_trials} attempts.")

        uri = f"{args.prefix}_{len(mixtures):08d}"
        record = build_candidate_mixture(
            uri=uri,
            groups=groups,
            rng=rng,
            overlap_center=overlap_center,
            overlap_offset=overlap_offset,
            min_mixture_duration=args.min_mixture_duration,
            max_mixture_duration=args.max_mixture_duration,
            max_speakers=args.max_speakers,
            max_speakers_per_frame=args.max_speakers_per_frame,
        )
        if record is not None:
            mixtures.append(record)

    write_json(mixtures, args.out, force=args.force)
    print(f"Loaded active segments: {sum(len(group.segments) for group in groups)}")
    print(f"Loaded speaker groups: {len(groups)}")
    print(f"Wrote mixtures: {len(mixtures)}")
    print(f"Attempts: {attempts}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()

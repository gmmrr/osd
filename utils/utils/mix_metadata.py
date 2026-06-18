#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from config.params import AUDIO_SAMPLE_RATE

DEFAULT_MAX_NUMBER_SPEAKER = 3
DEFAULT_MAX_OVERLAP_SPEAKER = 2
MAX_FINAL_SEGMENT_TRIES = 5


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
        (
            text == name,
            text == stem,
            text == base,
            path_value.name == name,
            path_value.stem in {stem, base},
            resolved == text,
            text.endswith(name),
            text.endswith(stem),
            text.endswith(base),
        )
    )


def _speaker_key(seg: ActiveSegment) -> str:
    return seg.speaker_id if seg.speaker_id not in (None, "") else seg.audio_derivative


def _local_speaker_label(uri: str, seg: ActiveSegment, speaker_map: dict[str, str]) -> str:
    key = _speaker_key(seg)
    if key not in speaker_map:
        speaker_map[key] = f"{uri}_spk{len(speaker_map) + 1}"
    return speaker_map[key]


def _can_append_segment(
    seg: ActiveSegment,
    prev: ActiveSegment,
    seen_speakers: set[str],
    speaker_aware: bool,
    max_number_speaker: int,
) -> bool:
    if seg.audio_derivative == prev.audio_derivative:
        return False

    speaker = seg.speaker_id
    if not speaker_aware:
        return True

    if speaker in (None, ""):
        return False

    return speaker in seen_speakers or len(seen_speakers) < max_number_speaker


def _speaker_id_text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _row_matches_audio(audio_path: Path, row: dict[str, Any]) -> bool:
    return _match_audio_ref(audio_path, row.get("wav_path")) or _match_audio_ref(audio_path, row.get("input"))


def _speaker_id_from_metadata_file(audio_path: Path, meta_path: Path) -> str | None:
    try:
        if meta_path.suffix == ".jsonl":
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and (row := json.loads(line)) and isinstance(row, dict) and _row_matches_audio(audio_path, row):
                        return _speaker_id_text(row.get("speaker_id"))
        else:
            with meta_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if _row_matches_audio(audio_path, row):
                        return _speaker_id_text(row.get("speaker_id"))
    except Exception:
        return None
    return None


def infer_speaker_id(audio_path: Path) -> str | None:
    for parent in [audio_path.parent, *audio_path.parents]:
        for meta_path in (parent / "metadata.jsonl", parent / "metadata.csv"):
            if meta_path.exists():
                speaker_id = _speaker_id_from_metadata_file(audio_path, meta_path)
                if speaker_id is not None:
                    return speaker_id
    return None


def _resolve_speaker_id(item: dict[str, Any], audio_path: Path) -> str | None:
    speaker_id = item.get("speaker_id")
    if speaker_id in (None, "") and isinstance(item.get("parameters"), dict):
        speaker_id = item["parameters"].get("speaker_id")
    if speaker_id in (None, ""):
        speaker_id = infer_speaker_id(audio_path)
    return _speaker_id_text(speaker_id)


def _parse_vad_segment(audio: str, audio_derivative: str, speaker_id: str | None, index: int, seg: dict[str, Any]) -> ActiveSegment | None:
    try:
        start = float(seg["start"])
        end = float(seg["end"])
    except Exception:
        return None
    if end <= start:
        return None
    return ActiveSegment(
        source_id=f"{audio_derivative}_seg{index:04d}",
        audio=audio,
        audio_derivative=audio_derivative,
        speaker_id=speaker_id,
        segment_index=index,
        orig_start=start,
        orig_end=end,
    )


def load_vad_json(path: Path, require_speaker_id: bool = False) -> list[ActiveSegment]:
    with path.open("r", encoding="utf-8") as f:
        item = json.load(f)

    if "segments" not in item or "input" not in item:
        return []

    sr = int(item.get("sampling_rate", AUDIO_SAMPLE_RATE))
    if sr != AUDIO_SAMPLE_RATE:
        raise ValueError(f"Unexpected sampling_rate={sr} in {path}; expected {AUDIO_SAMPLE_RATE}.")

    audio = item["input"]
    audio_derivative = item.get("audio_derivative", Path(audio).stem)
    speaker_id = _resolve_speaker_id(item, Path(audio))
    if require_speaker_id and speaker_id in (None, ""):
        raise ValueError(f"speaker_id is null for {path}")

    return [
        segment
        for i, seg in enumerate(item.get("segments", []))
        if (segment := _parse_vad_segment(audio, audio_derivative, speaker_id, i, seg)) is not None
    ]


def collect_segments(vad_root: Path, min_duration: float, require_speaker_id: bool = False) -> list[ActiveSegment]:
    segments: list[ActiveSegment] = []
    for path in sorted(vad_root.rglob("*.json")):
        segments.extend(
            seg for seg in load_vad_json(path, require_speaker_id=require_speaker_id) if seg.duration >= min_duration
        )
    return segments


def sample_overlap_ratio(rng: random.Random, center: float, offset: float) -> float:
    low = max(0.0, center - offset)
    high = min(1.0, center + offset)
    return rng.uniform(low, high)


def choose_chain(
    segments: list[ActiveSegment],
    rng: random.Random,
    overlap_center: float,
    overlap_offset: float,
    max_number_speaker: int,
    max_overlap_speaker: int,
    min_mixture_duration: float,
    max_mixture_duration: float,
    speaker_aware: bool = True,
    pre_silence_range: tuple[float, float] = (0.0, 0.3),
    max_trials: int = 1000,
) -> tuple[list[ActiveSegment], list[float]] | None:
    if len(segments) < 2:
        return None

    if speaker_aware:
        segments = [seg for seg in segments if seg.speaker_id not in (None, "")]
        if len(segments) < 2:
            return None

    for _ in range(max_trials):
        mix_start = rng.uniform(*pre_silence_range)
        chain = [rng.choice(segments)]
        used = {chain[0].audio_derivative}
        ratios: list[float] = []
        ends = [mix_start + chain[0].duration]
        current_end = ends[0]
        seen_speakers = {chain[0].speaker_id} if chain[0].speaker_id not in (None, "") else set()

        while current_end < min_mixture_duration:
            if not speaker_aware and len(chain) >= max_number_speaker:
                break

            prev = chain[-1]
            choices = [
                seg
                for seg in segments
                if seg.audio_derivative not in used
                and _can_append_segment(seg, prev, seen_speakers, speaker_aware, max_number_speaker)
            ]

            if not choices:
                break

            picked: ActiveSegment | None = None
            picked_end: float | None = None
            picked_ratio: float | None = None
            for _ in range(MAX_FINAL_SEGMENT_TRIES):
                seg = rng.choice(choices)
                ratio = sample_overlap_ratio(rng, overlap_center, overlap_offset)
                overlap = ratio * min(prev.duration, seg.duration)
                mix_start = current_end - overlap
                mix_end = mix_start + seg.duration
                if mix_end <= max_mixture_duration and sum(end > mix_start for end in ends) < max_overlap_speaker:
                    picked = seg
                    picked_end = mix_end
                    picked_ratio = ratio
                    break

            if picked is None:
                break

            chain.append(picked)
            used.add(picked.audio_derivative)
            ratios.append(float(picked_ratio))
            ends.append(float(picked_end))
            if picked.speaker_id not in (None, ""):
                seen_speakers.add(picked.speaker_id)
            current_end = float(picked_end)

        if min_mixture_duration <= current_end <= max_mixture_duration:
            if len(seen_speakers) > max_number_speaker:
                continue
            return chain, ratios

    return None


def place_sources(
    uri: str,
    chain: list[ActiveSegment],
    ratios: list[float],
    rng: random.Random,
    pre_silence_range: tuple[float, float] = (0.0, 0.3),
    gain_db_range: tuple[float, float] = (-3.0, 3.0),
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    speaker_map: dict[str, str] = {}

    mix_start = rng.uniform(*pre_silence_range)
    mix_end = mix_start + chain[0].duration

    for idx, seg in enumerate(chain, start=1):
        if idx > 1:
            overlap = ratios[idx - 2] * min(chain[idx - 2].duration, seg.duration)
            mix_start = mix_end - overlap
            mix_end = mix_start + seg.duration

        sources.append(
            {
                "source": f"s{idx}",
                "source_id": seg.source_id,
                "audio": seg.audio,
                "audio_derivative": seg.audio_derivative,
                "speaker_id": seg.speaker_id,
                "segment_index": seg.segment_index,
                "speaker": _local_speaker_label(uri, seg, speaker_map),
                "orig_start": round(seg.orig_start, 6),
                "orig_end": round(seg.orig_end, 6),
                "orig_duration": round(seg.duration, 6),
                "mix_start": round(mix_start, 6),
                "mix_end": round(mix_end, 6),
                "mix_duration": round(seg.duration, 6),
                "gain_db": round(rng.uniform(*gain_db_range), 3),
            }
        )

    return sources


def build_overlap_records(sources: list[dict[str, Any]], ratios: list[float]) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for idx in range(1, len(sources)):
        prev = sources[idx - 1]
        curr = sources[idx]
        start = max(float(prev["mix_start"]), float(curr["mix_start"]))
        end = min(float(prev["mix_end"]), float(curr["mix_end"]))
        dur = max(0.0, end - start)
        # VAD segments already represent voiced spans, so silence is excluded here.
        denom = min(float(prev["orig_duration"]), float(curr["orig_duration"]))
        overlaps.append(
            {
                "between": [prev["source"], curr["source"]],
                "start": round(start, 6),
                "end": round(end, 6),
                "duration": round(dur, 6),
                "ratio_target": ratios[idx - 1] if idx - 1 < len(ratios) else None,
                "ratio_actual": round(dur / denom, 6) if denom > 0 else None,
            }
        )
    return overlaps


def build_mixture_metadata(
    uri: str,
    segments: list[ActiveSegment],
    overlap_center: float,
    overlap_offset: float,
    max_number_speaker: int,
    max_overlap_speaker: int,
    speaker_aware: bool,
    rng: random.Random,
    min_mixture_duration: float,
    max_mixture_duration: float,
) -> dict[str, Any] | None:
    chosen = choose_chain(
        segments,
        rng,
        overlap_center,
        overlap_offset,
        max_number_speaker,
        max_overlap_speaker,
        min_mixture_duration,
        max_mixture_duration,
        speaker_aware=speaker_aware,
    )
    if chosen is None:
        return None

    chain, ratios = chosen
    sources = place_sources(uri, chain, ratios, rng)
    speaker_labels = sorted({src["speaker"] for src in sources})
    if len(speaker_labels) > max_number_speaker:
        return None

    natural_duration = max(float(src["mix_end"]) for src in sources)
    if not (min_mixture_duration <= natural_duration <= max_mixture_duration):
        return None

    overlaps = build_overlap_records(sources, ratios)
    speaker_ids = sorted({src["speaker_id"] for src in sources if src["speaker_id"] not in (None, "")})
    return {
        "uri": uri,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "duration": round(natural_duration, 6),
        "natural_duration": round(natural_duration, 6),
        "speaker_count": len(speaker_labels),
        "speaker_labels": speaker_labels,
        "max_number_speaker": max_number_speaker,
        "max_overlap_speaker": max_overlap_speaker,
        "overlap_ratio_center": overlap_center,
        "overlap_ratio_offset": overlap_offset,
        "speaker_aware": speaker_aware,
        "source_count": len(sources),
        "speaker_ids": speaker_ids,
        "overlap_ratio_targets": ratios,
        "overlap_ratio_target": ratios[0] if ratios else None,
        "overlap_ratio_actuals": [ovr["ratio_actual"] for ovr in overlaps],
        "overlap_ratio_actual": overlaps[0]["ratio_actual"] if overlaps else None,
        "overlap": overlaps,
        "sources": sources,
    }


def write_jsonl(items: list[dict[str, Any]], path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VAD-based mixture metadata.")
    parser.add_argument("--vad-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-mixtures", type=int, default=5)
    parser.add_argument(
        "--max-number-speaker",
        type=int,
        default=DEFAULT_MAX_NUMBER_SPEAKER,
        help="Maximum number of distinct speakers allowed in one mixture.",
    )
    parser.add_argument(
        "--max-overlap-speaker",
        type=int,
        default=DEFAULT_MAX_OVERLAP_SPEAKER,
        help="Maximum number of simultaneously overlapping speakers.",
    )
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
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it already exists.")
    parser.add_argument(
        "--disallow-anonymous",
        action="store_true",
        help="Require non-empty speaker_id values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_mixture_duration <= 0:
        raise ValueError("--min-mixture-duration must be positive.")
    if args.max_mixture_duration < args.min_mixture_duration:
        raise ValueError("--max-mixture-duration must be >= --min-mixture-duration.")
    if args.max_number_speaker <= 0:
        raise ValueError("--max-number-speaker must be positive.")
    if args.max_overlap_speaker <= 0:
        raise ValueError("--max-overlap-speaker must be positive.")
    overlap_center, overlap_offset = args.overlap_ratios
    if overlap_offset < 0:
        raise ValueError("--overlap-ratios OFFSET must be non-negative.")

    rng = random.Random(args.seed)
    segments = collect_segments(args.vad_dir, args.min_duration, require_speaker_id=args.disallow_anonymous)
    if len(segments) < 2:
        raise RuntimeError("Need at least two valid active segments.")
    speaker_aware = all(seg.speaker_id not in (None, "") for seg in segments)

    mixtures: list[dict[str, Any]] = []
    attempts = 0
    while len(mixtures) < args.n_mixtures:
        attempts += 1
        if attempts > args.max_build_trials:
            raise RuntimeError(
                f"Only generated {len(mixtures)} mixtures after {args.max_build_trials} attempts."
            )

        metadata = build_mixture_metadata(
            uri=f"{args.prefix}_{len(mixtures):08d}",
            segments=segments,
            overlap_center=overlap_center,
            overlap_offset=overlap_offset,
            max_number_speaker=args.max_number_speaker,
            max_overlap_speaker=args.max_overlap_speaker,
            speaker_aware=speaker_aware,
            rng=rng,
            min_mixture_duration=args.min_mixture_duration,
            max_mixture_duration=args.max_mixture_duration,
        )
        if metadata is not None:
            mixtures.append(metadata)

    write_jsonl(mixtures, args.out, force=args.force)
    print(f"Loaded active segments: {len(segments)}")
    print(f"Wrote mixtures: {len(mixtures)}")
    print(f"Attempts: {attempts}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()

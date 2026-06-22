#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    from config.constants import EXT_WAV
    from config.params import AUDIO_SAMPLE_RATE
except ModuleNotFoundError:  # pragma: no cover - convenience for `uv run utils/mix.py`
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config.constants import EXT_WAV
    from config.params import AUDIO_SAMPLE_RATE


SPLITS = ("train", "dev", "test")


def load_mix_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"Expected JSON array or object in {path}, got {type(data).__name__}.")


def load_source_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if sr != sample_rate:
        raise ValueError(f"Unexpected sample rate for {path}: {sr} != {sample_rate}.")
    if audio.shape[1] != 1:
        raise ValueError(f"Expected mono audio for {path}, got {audio.shape[1]} channels.")
    return audio.T


def slice_audio(audio: np.ndarray, start_sec: float, end_sec: float, sample_rate: int) -> np.ndarray:
    start = max(0, int(round(start_sec * sample_rate)))
    end = max(start, int(round(end_sec * sample_rate)))
    end = min(end, audio.shape[1])
    if start >= end:
        return audio[:, :0]
    return audio[:, start:end]


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    return audio * float(10 ** (gain_db / 20))


def render_mix(record: dict[str, Any], output_path: Path) -> Path:
    sample_rate = int(record.get("sample_rate", AUDIO_SAMPLE_RATE))
    duration = float(record["duration"])
    out_frames = int(round(duration * sample_rate))

    sources = record.get("sources") or []
    if not sources:
        raise ValueError(f"Record {record.get('uri', '?')} has no sources.")

    mix = np.zeros((1, out_frames), dtype=np.float32)
    for source in sources:
        audio_path = Path(source["audio"])
        audio = load_source_audio(audio_path, sample_rate)
        segment = slice_audio(audio, float(source["orig_start"]), float(source["orig_end"]), sample_rate)
        segment = apply_gain(segment, float(source.get("gain_db", 0.0)))

        start = max(0, int(round(float(source["mix_start"]) * sample_rate)))
        end = min(out_frames, start + segment.shape[1])
        if end <= start:
            continue

        mix[:, start:end] += segment[:, : end - start]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mix.T, sample_rate, subtype="FLOAT")
    return output_path


def split_key(record: dict[str, Any], train_ratio: float, dev_ratio: float, test_ratio: float) -> str:
    raw_split = str(record.get("split", "")).strip().lower()
    if raw_split in {"train", "dev", "test"}:
        return raw_split
    if raw_split in {"val", "valid", "validation"}:
        return "dev"

    total = train_ratio + dev_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    digest = hashlib.sha1(str(record.get("uri", "")).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)

    train_cut = train_ratio / total
    dev_cut = (train_ratio + dev_ratio) / total
    if value < train_cut:
        return "train"
    if value < dev_cut:
        return "dev"
    return "test"


def format_rttm_line(uri: str, speaker: str, start: float, end: float) -> str:
    duration = max(0.0, end - start)
    return f"SPEAKER {uri} 1 {start:.6f} {duration:.6f} <NA> <NA> {speaker} <NA> <NA>"


def format_uem_line(uri: str, duration: float) -> str:
    return f"{uri} 1 0.000000 {duration:.6f}"


def write_text(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_split_files(dataset_root: Path, records: list[dict[str, Any]], force: bool) -> None:
    audio_dir = dataset_root / "audio"
    rttm_dir = dataset_root / "rttm"
    uem_dir = dataset_root / "uem"
    lists_dir = dataset_root / "lists"

    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for record in records:
        split = split_key(record, 0.8, 0.1, 0.1)
        split_records[split].append(record)

        uri = str(record.get("uri", "mix"))
        audio_path = audio_dir / f"{uri}{EXT_WAV}"
        if not audio_path.exists() or force:
            render_mix(record, audio_path)

    for split, items in split_records.items():
        rttm_lines: list[str] = []
        uem_lines: list[str] = []
        list_lines: list[str] = []

        for record in items:
            uri = str(record.get("uri", "mix"))
            duration = float(record["duration"])
            list_lines.append(uri)
            uem_lines.append(format_uem_line(uri, duration))
            for source in record.get("sources", []):
                try:
                    rttm_lines.append(
                        format_rttm_line(
                            uri=uri,
                            speaker=str(source["speaker"]),
                            start=float(source["mix_start"]),
                            end=float(source["mix_end"]),
                        )
                    )
                except Exception:
                    continue

        write_text(rttm_dir / f"{split}.rttm", "\n".join(rttm_lines) + ("\n" if rttm_lines else ""), force)
        write_text(uem_dir / f"{split}.uem", "\n".join(uem_lines) + ("\n" if uem_lines else ""), force)
        write_text(lists_dir / f"{split}.lst", "\n".join(list_lines) + ("\n" if list_lines else ""), force)

    database_yaml = "\n".join(
        [
            "dataset:",
            "  root: .",
            "  audio: audio/{uri}.wav",
            "  splits:",
            "    train:",
            "      list: lists/train.lst",
            "      rttm: rttm/train.rttm",
            "      uem: uem/train.uem",
            "    dev:",
            "      list: lists/dev.lst",
            "      rttm: rttm/dev.rttm",
            "      uem: uem/dev.uem",
            "    test:",
            "      list: lists/test.lst",
            "      rttm: rttm/test.rttm",
            "      uem: uem/test.uem",
        ]
    )
    write_text(dataset_root / "database.yml", database_yaml + "\n", force)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render mixture wavs and dataset manifests from mix metadata JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Path to mix.json produced by mix_metadata.py.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dataset root directory to write audio/, rttm/, uem/, lists/, and database.yml.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_mix_records(args.input)
    if not records:
        raise RuntimeError(f"No mix records found in {args.input}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split_files(args.output_dir, records, force=args.force)

    print(f"Loaded records: {len(records)}")
    print(f"Output root: {args.output_dir}")
    print("Wrote: audio/, rttm/, uem/, lists/, database.yml")


if __name__ == "__main__":
    main()

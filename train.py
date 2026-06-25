#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch


SPLITS = ("train", "development", "test")
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_DATABASE_NAME = "MixDataset"
DEFAULT_PROTOCOL_NAME = "MixDiarization"
DEFAULT_MODEL_DIR = Path("models/pyannote-segmentation-3.0")
DEFAULT_CHUNK_DURATION = 10.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_EPOCHS = 1
DEFAULT_PRETRAINED_STRIDE = 10


def load_mix_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"Expected JSON object or array in {path}, got {type(data).__name__}.")


def split_name(record: dict[str, Any], ratios: tuple[float, float, float]) -> str:
    raw_split = str(record.get("split", "")).strip().lower()
    if raw_split in {"train", "development", "dev", "valid", "validation", "test"}:
        return "development" if raw_split in {"dev", "valid", "validation"} else raw_split

    total = sum(ratios)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    digest = hashlib.sha1(str(record.get("uri", "")).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    train_cut = ratios[0] / total
    dev_cut = (ratios[0] + ratios[1]) / total
    if value < train_cut:
        return "train"
    if value < dev_cut:
        return "development"
    return "test"


def ensure_audio_file(source: Path, target: Path, copy_audio: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return

    if copy_audio:
        shutil.copy2(source, target)
        return

    try:
        target.symlink_to(source.resolve())
    except Exception:
        shutil.copy2(source, target)


def format_rttm_line(uri: str, speaker: str, start: float, end: float) -> str:
    duration = max(0.0, end - start)
    return f"SPEAKER {uri} 1 {start:.6f} {duration:.6f} <NA> <NA> {speaker} <NA> <NA>"


def format_uem_line(uri: str, duration: float) -> str:
    return f"{uri} NA 0.000000 {duration:.6f}"


def build_workspace(
    mix_json: Path,
    audio_dir: Path,
    output_dir: Path,
    copy_audio: bool,
    split_ratios: tuple[float, float, float],
    database_name: str,
    protocol_name: str,
) -> dict[str, list[dict[str, Any]]]:
    records = load_mix_records(mix_json)
    if not records:
        raise RuntimeError(f"No mix records found in {mix_json}.")

    audio_out = output_dir / "audio"
    rttm_out = output_dir / "rttm"
    uem_out = output_dir / "uem"
    lists_out = output_dir / "lists"
    for path in (audio_out, rttm_out, uem_out, lists_out):
        path.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    rttm_lines: dict[str, list[str]] = {split: [] for split in SPLITS}
    uem_lines: dict[str, list[str]] = {split: [] for split in SPLITS}
    list_lines: dict[str, list[str]] = {split: [] for split in SPLITS}

    for record in records:
        uri = str(record.get("uri"))
        if not uri:
            raise ValueError("Each mix record must contain a non-empty 'uri'.")

        split = split_name(record, split_ratios)
        split_records[split].append(record)
        list_lines[split].append(uri)

        source_audio = audio_dir / f"{uri}.wav"
        if not source_audio.exists():
            raise FileNotFoundError(f"Missing mixed audio for {uri}: {source_audio}")
        ensure_audio_file(source_audio, audio_out / f"{uri}.wav", copy_audio=copy_audio)

        duration = float(record["duration"])
        uem_lines[split].append(format_uem_line(uri, duration))

        for source in record.get("sources", []):
            try:
                rttm_lines[split].append(
                    format_rttm_line(
                        uri=uri,
                        speaker=str(source["speaker"]),
                        start=float(source["mix_start"]),
                        end=float(source["mix_end"]),
                    )
                )
            except Exception:
                continue

    for split in SPLITS:
        (lists_out / f"{split}.lst").write_text("\n".join(list_lines[split]) + ("\n" if list_lines[split] else ""), encoding="utf-8")
        (rttm_out / f"{split}.rttm").write_text("\n".join(rttm_lines[split]) + ("\n" if rttm_lines[split] else ""), encoding="utf-8")
        (uem_out / f"{split}.uem").write_text("\n".join(uem_lines[split]) + ("\n" if uem_lines[split] else ""), encoding="utf-8")

    database_yml = "\n".join(
        [
            "Databases:",
            f"  {database_name}:",
            "    - audio/{uri}.wav",
            "Protocols:",
            f"  {database_name}:",
            "    SpeakerDiarization:",
            f"      {protocol_name}:",
            "        scope: file",
            "        train:",
            "          uri: lists/train.lst",
            "          annotation: rttm/train.rttm",
            "          annotated: uem/train.uem",
            "        development:",
            "          uri: lists/development.lst",
            "          annotation: rttm/development.rttm",
            "          annotated: uem/development.uem",
            "        test:",
            "          uri: lists/test.lst",
            "          annotation: rttm/test.rttm",
            "          annotated: uem/test.uem",
        ]
    )
    (output_dir / "database.yml").write_text(database_yml + "\n", encoding="utf-8")
    return split_records


def load_pretrained_model(model_dir: Path, task) -> Any:
    from pyannote.audio.models.segmentation import PyanNet

    model = PyanNet(
        task=task,
        sincnet={"stride": DEFAULT_PRETRAINED_STRIDE},
        lstm={"hidden_size": 128, "num_layers": 4, "bidirectional": True, "monolithic": True},
        linear={"hidden_size": 128, "num_layers": 2},
    )

    checkpoint = model_dir / "pytorch_model.bin"
    if checkpoint.exists():
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model


def train_model(
    workspace: Path,
    model_dir: Path,
    database_name: str,
    protocol_name: str,
    chunk_duration: float,
    batch_size: int,
    max_speakers_per_chunk: int,
    max_speakers_per_frame: int,
    max_epochs: int,
    device: str,
) -> Path:
    from pyannote.database import registry
    from pyannote.audio.tasks import SpeakerDiarization
    import pytorch_lightning as pl

    registry.load_database(str(workspace / "database.yml"))
    protocol = registry.get_protocol(f"{database_name}.SpeakerDiarization.{protocol_name}")

    task = SpeakerDiarization(
        protocol,
        duration=chunk_duration,
        batch_size=batch_size,
        max_speakers_per_chunk=max_speakers_per_chunk,
        max_speakers_per_frame=max_speakers_per_frame,
    )

    model = load_pretrained_model(model_dir, task)
    trainer = pl.Trainer(
        accelerator="auto" if device == "auto" else device,
        devices=1,
        max_epochs=max_epochs,
        default_root_dir=str(workspace / "checkpoints"),
    )
    trainer.fit(model)

    checkpoint_dir = workspace / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pyannote segmentation-3.0 on mixed diarization data.")
    parser.add_argument("--mix-json", type=Path, required=True, help="Path to mix.json produced by mix_metadata.py.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing mixed wav files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training workspace / dataset root.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local pyannote segmentation-3.0 directory.")
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=DEFAULT_SPLIT_RATIOS,
        metavar=("TRAIN", "DEV", "TEST"),
        help="Deterministic split ratios used when mix.json has no split column.",
    )
    parser.add_argument("--database-name", type=str, default=DEFAULT_DATABASE_NAME)
    parser.add_argument("--protocol-name", type=str, default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--chunk-duration", type=float, default=DEFAULT_CHUNK_DURATION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--max-speakers-per-chunk", type=int, default=3)
    parser.add_argument("--max-speakers-per-frame", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--copy-audio", action="store_true", help="Copy audio instead of symlinking it into the workspace.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_ratios = tuple(float(value) for value in args.split_ratios)
    if len(split_ratios) != 3:
        raise ValueError("--split-ratios must contain exactly three values.")
    if sum(split_ratios) <= 0:
        raise ValueError("--split-ratios must sum to a positive value.")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Local model directory not found: {args.model_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_records = build_workspace(
        mix_json=args.mix_json,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        copy_audio=args.copy_audio,
        split_ratios=split_ratios,
        database_name=args.database_name,
        protocol_name=args.protocol_name,
    )

    print(f"Workspace: {args.output_dir}")
    for split in SPLITS:
        print(f"{split}: {len(split_records[split])} record(s)")

    checkpoint_dir = train_model(
        workspace=args.output_dir,
        model_dir=args.model_dir,
        database_name=args.database_name,
        protocol_name=args.protocol_name,
        chunk_duration=args.chunk_duration,
        batch_size=args.batch_size,
        max_speakers_per_chunk=args.max_speakers_per_chunk,
        max_speakers_per_frame=args.max_speakers_per_frame,
        max_epochs=args.max_epochs,
        device=args.device,
    )

    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- VAD ---

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.params import (
    AUDIO_SAMPLE_RATE,
    VAD_ENERGY_REL_THRESHOLD,
    VAD_SMOOTHING_WINDOW,
    VAD_EXPAND_PRE,
    VAD_EXPAND_POST,
    VAD_EXPAND_DELTA,
)

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _load_audio(path: Path, sample_rate: int = AUDIO_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1)
    if sr != sample_rate:
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return wav.astype(np.float32, copy=False), sr


def load_vad_model(device: str = "auto"):
    used_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    vad_model = load_silero_vad()
    try:
        vad_model.to(used_device)
    except Exception:
        pass
    vad_model.eval()
    return vad_model, used_device


def vad_single_audio(
    audio_path: Path,
    vad_model,
    device: str,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_pad_ms: int,
    force: bool = False,
) -> Path:
    """
    Run VAD on one normalized file.

    Expected input:
        <stem>_std_nml.wav
    Output:
        <stem>_std_nml_vad.json
    """
    audio_path = Path(audio_path)
    out_path = audio_path.with_name(f"{audio_path.stem}_vad.json")
    if out_path.exists() and not force:
        print(f"↪ {audio_path.name}: VAD json already exists (cached)")
        return out_path

    wav_cpu, sr = _load_audio(audio_path)
    wav = torch.from_numpy(np.asarray(wav_cpu, dtype=np.float32)).unsqueeze(0)  # type: ignore[name-defined]

    try:
        wav_dev = wav.to(device)
    except Exception:
        wav_dev = wav

    speech_timestamps = get_speech_timestamps(
        wav_dev,
        vad_model,
        sampling_rate=sr,
        threshold=vad_threshold,
        min_speech_duration_ms=vad_min_speech_ms,
        min_silence_duration_ms=vad_min_silence_ms,
        speech_pad_ms=vad_pad_ms,
    )

    energy = np.abs(wav_cpu)
    smooth = np.convolve(energy, np.ones(VAD_SMOOTHING_WINDOW) / VAD_SMOOTHING_WINDOW, mode="same")
    mean_e = float(np.mean(smooth)) if smooth.size else 0.0
    energy_threshold = VAD_ENERGY_REL_THRESHOLD * mean_e
    n_samples = len(wav_cpu)

    for seg in speech_timestamps:
        start = int(seg["start"])
        end = int(seg["end"])
        start = max(0, start - int(VAD_EXPAND_PRE * sr))
        end = min(n_samples, end + int(VAD_EXPAND_POST * sr))
        step_delta = max(1, int(VAD_EXPAND_DELTA * sr))

        while start > 0 and smooth[start] > energy_threshold:
            start = max(0, start - step_delta)
        while end < n_samples - 1 and smooth[end] > energy_threshold:
            end = min(n_samples - 1, end + step_delta)

        seg["start_s"] = round(start / sr, 6)
        seg["end_s"] = round(end / sr, 6)

    segments = [{"start": seg["start_s"], "end": seg["end_s"]} for seg in speech_timestamps]

    payload = {
        "input": str(audio_path.resolve()),
        "audio_derivative": audio_path.stem,
        "sampling_rate": sr,
        "segment_count": len(segments),
        "segments": segments,
        "parameters": {
            "model": "Silero-VAD",
            "threshold": vad_threshold,
            "pad_ms": vad_pad_ms,
            "min_speech_ms": vad_min_speech_ms,
            "min_silence_ms": vad_min_silence_ms,
            "energy_threshold_rel": VAD_ENERGY_REL_THRESHOLD,
            "smoothing_window": VAD_SMOOTHING_WINDOW,
            "expand_pre_s": VAD_EXPAND_PRE,
            "expand_post_s": VAD_EXPAND_POST,
            "expand_delta_s": VAD_EXPAND_DELTA,
            "device": device,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"VAD: {audio_path.name} -> {out_path.name}")
    return out_path


def run_vad_dir(
    input_dir: Path | str,
    audio_pattern: str = "*_std_nml.wav",
    vad_threshold: float = 0.20,
    vad_min_speech_ms: int = 75,
    vad_min_silence_ms: int = 75,
    vad_pad_ms: int = 50,
    device: str = "auto",
    force: bool = False,
) -> None:
    """
    Run VAD on all normalized files in a directory.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_files = sorted(
        p for p in input_dir.glob(audio_pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
    if not input_files:
        print(f"⚠️  No normalized audio files found in: {input_dir}")
        return

    vad_model, used_device = load_vad_model(device)
    print(f"🚀 VAD in '{input_dir}'")
    print(f"   • Device: {used_device}")
    print(f"   • Files: {len(input_files)}")
    print(f"   • Pattern: {audio_pattern}")

    for file in input_files:
        vad_single_audio(
            file,
            vad_model=vad_model,
            device=used_device,
            vad_threshold=vad_threshold,
            vad_min_speech_ms=vad_min_speech_ms,
            vad_min_silence_ms=vad_min_silence_ms,
            vad_pad_ms=vad_pad_ms,
            force=force,
        )

    print("✅ VAD completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VAD over a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_std_nml.wav files")
    parser.add_argument(
        "--audio-pattern",
        default="*_std_nml.wav",
        help="Glob pattern used to select input files",
    )
    parser.add_argument("--threshold", type=float, default=0.20, help="Silero speech probability threshold")
    parser.add_argument("--min_speech_ms", type=int, default=75, help="Minimum speech segment length (ms)")
    parser.add_argument("--min_silence_ms", type=int, default=75, help="Minimum silence gap (ms)")
    parser.add_argument("--pad_ms", type=int, default=50, help="Context padding around each segment (ms)")
    parser.add_argument("--device", default="auto", help="Device to use: auto | cuda | cpu")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    run_vad_dir(
        args.input_dir,
        audio_pattern=args.audio_pattern,
        vad_threshold=args.threshold,
        vad_min_speech_ms=args.min_speech_ms,
        vad_min_silence_ms=args.min_silence_ms,
        vad_pad_ms=args.pad_ms,
        device=args.device,
        force=args.force,
    )

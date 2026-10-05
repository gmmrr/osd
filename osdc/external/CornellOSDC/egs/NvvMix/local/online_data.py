from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


AMI_ONLINE_DATA = Path(__file__).resolve().parents[2] / "AMI" / "local" / "online_data.py"
spec = importlib.util.spec_from_file_location("cornell_ami_online_data", AMI_ONLINE_DATA)
ami_online_data = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ami_online_data)


class OnlineFeats(ami_online_data.OnlineFeats):
    """Cornell feature extraction backed by NvvMix source manifests."""

    def __init__(self, audio_root, label_root, split, configs, segment=300, probs=None):
        if probs:
            raise ValueError("NvvMix uses pre-mixed audio; online AMI mixing is disabled.")

        self.configs = configs
        self.segment = segment
        self.probs = None
        audio_root = Path(audio_root).resolve()
        label_root = Path(label_root).resolve()
        manifests = sorted(label_root.glob(f"*/manifests/{split}.json"))
        if not manifests:
            raise FileNotFoundError(f"No {split} manifests found under {label_root}")

        self.records = []
        for manifest in manifests:
            self.records.extend(json.loads(manifest.read_text(encoding="utf-8")))
        self.records = [
            record
            for record in self.records
            if audio_root in Path(record["audio"]).resolve().parents
            and len(sf.SoundFile(record["label"])) > self.segment + 2
        ]
        if not self.records:
            raise ValueError(f"No usable {split} recordings under {label_root}")

        self.tot_length = sum(
            len(sf.SoundFile(record["label"])) for record in self.records
        ) // self.segment
        self.set_feats_func()
        print(
            f"OnlineFeats[{split}]: {len(self.records)} recordings from "
            f"{len(manifests)} NvvMix sources."
        )

    def noaugm(self):
        record = random.choice(self.records)
        label_file = record["label"]
        audio_file = record["audio"]
        label_length = len(sf.SoundFile(label_file))
        start = np.random.randint(0, label_length - self.segment - 2)
        stop = start + self.segment
        labels, _ = sf.read(label_file, start=start, stop=stop)

        sample_rate = self.configs["data"]["fs"]
        hop_size = self.configs["feats"]["hop_size"]
        first_sample = int(start * sample_rate * hop_size)
        last_sample = int(stop * sample_rate * hop_size)
        audio, _ = sf.read(audio_file, start=first_sample, stop=last_sample)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        features = self.feats_func(audio)
        labels = labels[: features.shape[-1]]
        return (
            features,
            torch.from_numpy(labels).long(),
            torch.ones(len(labels), dtype=torch.bool),
        )

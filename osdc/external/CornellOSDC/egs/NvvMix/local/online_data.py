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
        self.split = split
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

        if split == "train":
            # Preserve Cornell's online random sampling for training.
            self.examples = None
            self.tot_length = sum(
                len(sf.SoundFile(record["label"])) for record in self.records
            ) // self.segment
        else:
            # Validation and test must evaluate exactly the same windows on
            # every epoch.  Leaving sampling to ``noaugm`` would ignore the
            # DataLoader index and randomly change both the recording and the
            # segment start each time an item is requested.
            self.examples = []
            for record in self.records:
                label_length = len(sf.SoundFile(record["label"]))
                for start in range(0, label_length - self.segment - 2, self.segment):
                    self.examples.append((record, start))
            self.tot_length = len(self.examples)
        self.set_feats_func()
        print(
            f"OnlineFeats[{split}]: {len(self.records)} recordings, "
            f"{self.tot_length} {'random samples' if split == 'train' else 'fixed windows'} "
            f"from {len(manifests)} NvvMix sources."
        )

    def _load_segment(self, record, start):
        label_file = record["label"]
        audio_file = record["audio"]
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

    def noaugm(self):
        record = random.choice(self.records)
        label_length = len(sf.SoundFile(record["label"]))
        start = np.random.randint(0, label_length - self.segment - 2)
        return self._load_segment(record, start)

    def __getitem__(self, item):
        if self.split == "train":
            return self.noaugm()
        assert self.examples is not None
        record, start = self.examples[item]
        return self._load_segment(record, start)

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from online_data import OnlineFeats


AMI_TRAIN = Path(__file__).resolve().parents[2] / "AMI" / "local" / "train.py"
spec = importlib.util.spec_from_file_location("cornell_ami_train", AMI_TRAIN)
ami_train = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ami_train)


class OSDC_NVVMIX(ami_train.OSDC_AMI):
    def train_dataloader(self):
        return self._dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._dataloader("dev", shuffle=False)

    def _dataloader(self, split: str, shuffle: bool):
        dataset = OnlineFeats(
            self.configs["data"]["audio_root"],
            self.configs["data"]["label_root"],
            split,
            self.configs,
            segment=self.configs["data"]["segment"],
            probs=self.configs["augmentation"]["probs"],
        )
        return DataLoader(
            dataset,
            batch_size=self.configs["training"]["batch_size"],
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.configs["training"]["num_workers"],
            persistent_workers=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path)
    checkpoint.add_argument("--pretrained", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "confs.yml").write_text(
        yaml.safe_dump(configs), encoding="utf-8"
    )

    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="epoch{epoch:03d}-val{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=20, mode="min", verbose=True
    )
    logger = TensorBoardLogger(
        save_dir=os.fspath(args.output_dir.parent), name=args.output_dir.name
    )
    accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Accelerator:", accelerator)

    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=configs["training"]["n_epochs"],
        accumulate_grad_batches=configs["training"]["accumulate_batches"],
        gradient_clip_val=configs["training"]["gradient_clip"],
        callbacks=[checkpoint, early_stop],
        logger=logger,
    )
    model = OSDC_NVVMIX(configs)
    if args.pretrained:
        state = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"])

    trainer.fit(
        model,
        ckpt_path=os.fspath(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()

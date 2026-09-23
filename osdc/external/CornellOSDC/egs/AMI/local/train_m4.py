import os
import argparse
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from osdc.models.tcn import TCN
from online_data import OnlineFeats


parser = argparse.ArgumentParser()
parser.add_argument("conf_file", type=str)
parser.add_argument("log_dir", type=str)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Resume model, optimizer, scheduler, and epoch state from a Lightning checkpoint",
)


class OSDC_AMI(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        if not self.configs["augmentation"]["probs"]:
            weights = torch.tensor(
                [1.74, 1.0, 11.98, 219.0, 1000.0],
                dtype=torch.float32,
            )
        else:
            weights = torch.tensor(
                [1.0, 2.13, 6.89, 20.0, 115.0],
                dtype=torch.float32,
            )

        self.loss_fn = nn.CrossEntropyLoss(
            weight=weights,
            reduction="none",
        )

        self.model = TCN(
            in_chan=80,
            n_src=5,
            out_chan=1,
            n_blocks=5,
            n_repeats=3,
            bn_chan=64,
            hid_chan=128,
        )

    def forward(self, feats):
        return self.model(feats)

    def compute_loss(self, preds, labels, mask):
        loss = self.loss_fn(preds, labels)
        loss = loss * mask.detach()
        return loss.mean()

    def training_step(self, batch, batch_idx):
        feats, labels, mask = batch

        preds = self(feats)

        loss = self.compute_loss(
            preds,
            labels,
            mask,
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=feats.shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx):
        feats, labels, mask = batch

        preds = self(feats)

        loss = self.compute_loss(
            preds,
            labels,
            mask,
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=feats.shape[0],
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs["opt"]["lr"],
            weight_decay=self.configs["opt"]["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def train_dataloader(self):
        dataset = OnlineFeats(
            self.configs["data"]["chime6_root"],
            self.configs["data"]["label_train"],
            self.configs,
            probs=self.configs["augmentation"]["probs"],
            segment=self.configs["data"]["segment"],
        )

        return DataLoader(
            dataset,
            batch_size=self.configs["training"]["batch_size"],
            shuffle=True,
            num_workers=self.configs["training"]["num_workers"],
            drop_last=True,
        )

    def val_dataloader(self):
        dataset = OnlineFeats(
            self.configs["data"]["chime6_root"],
            self.configs["data"]["label_val"],
            self.configs,
            probs=None,
            segment=self.configs["data"]["segment"],
        )

        return DataLoader(
            dataset,
            batch_size=self.configs["training"]["batch_size"],
            shuffle=False,
            num_workers=self.configs["training"]["num_workers"],
            drop_last=True,
        )


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    os.makedirs(args.log_dir, exist_ok=True)

    with open(
        os.path.join(args.log_dir, "confs.yml"),
        "w",
    ) as f:
        yaml.safe_dump(configs, f)

    model = OSDC_AMI(configs)

    checkpoint_dir = os.path.join(
        args.log_dir,
        "checkpoints",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:03d}-val{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=True,
    )

    logger = TensorBoardLogger(
        save_dir=os.path.dirname(args.log_dir),
        name=os.path.basename(args.log_dir),
    )

    accelerator = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Accelerator:", accelerator)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=configs["training"]["n_epochs"],
        accumulate_grad_batches=configs["training"]["accumulate_batches"],
        gradient_clip_val=configs["training"]["gradient_clip"],
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
        ],
        logger=logger,
    )

    trainer.fit(model, ckpt_path=args.resume)

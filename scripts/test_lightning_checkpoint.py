#!/usr/bin/env python3
"""Run the test set of a trained checkpoint. This is the only entry point that evaluates the test split.

Training (audio_train.py / audio_train_resume.py) only validates. Usage:

    python scripts/test_lightning_checkpoint.py --conf_dir configs/lrs2_SwiftNet_6_freqmix_sum.yml \
        --ckpt_path Experiments/<exp>/checkpoints/last.ckpt --devices 1

The module is built in the same visual mode as the config (online or cached), the checkpoint is loaded
strictly, and trainer.test() runs test_step over the test dataloader.
"""
import argparse
import copy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pytorch_lightning as pl
import torch
import yaml

import look2hear.datas
import look2hear.losses
import look2hear.models
import look2hear.system
import look2hear.videomodels
from look2hear.utils.checkpoint_utils import check_architecture


def main(args):
    with open(args.conf_dir) as f:
        conf = yaml.safe_load(f)

    datamodule = getattr(look2hear.datas, conf["datamodule"]["data_name"])(**conf["datamodule"]["data_config"])
    datamodule.setup()
    _, _, test_loader = datamodule.make_loader

    audio = getattr(look2hear.models, conf["audionet"]["audionet_name"])(
        sample_rate=conf["datamodule"]["data_config"]["sample_rate"],
        **copy.deepcopy(conf["audionet"]["audionet_config"]),
    )
    visual_mode = "cached" if datamodule.uses_cached_visual else "online"
    video = None
    if visual_mode == "online":
        video = getattr(look2hear.videomodels, conf["videonet"]["videonet_name"])(
            **conf["videonet"]["videonet_config"]
        )

    loss = {
        split: getattr(look2hear.losses, cfg["loss_func"])(getattr(look2hear.losses, cfg["sdr_type"]), **cfg["config"])
        for split, cfg in (("train", conf["loss"]["train"]), ("val", conf["loss"]["val"]))
    }
    system = look2hear.system.AudioVisualLightningModule(
        audio_model=audio,
        video_model=video,
        loss_func=loss,
        optimizer=torch.optim.AdamW(audio.parameters(), lr=conf["optimizer"]["lr"]),
        test_loader=test_loader,
        config=conf,
        visual_mode=visual_mode,
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    check_architecture(ckpt, audio.architecture_summary())
    if visual_mode == "cached" and any(k.startswith("video_model.") for k in ckpt["state_dict"]):
        raise ValueError("Checkpoint has video_model weights but the config uses cached visual features.")
    if visual_mode == "online" and not any(k.startswith("video_model.") for k in ckpt["state_dict"]):
        raise ValueError("Checkpoint has no video_model weights: it was trained with cached visual features.")
    system.load_state_dict(ckpt["state_dict"], strict=True)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() and args.devices else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    results = trainer.test(system)
    print(results)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conf_dir", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--devices", type=int, default=1)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))

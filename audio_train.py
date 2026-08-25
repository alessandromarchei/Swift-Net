import os
import sys
import re
import torch
from torch import Tensor
import argparse
import json
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
import look2hear.videomodels
from look2hear.system import make_optimizer
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from rich import print, reconfigure
from collections.abc import MutableMapping
from look2hear.utils import (
    print_only,
    MyRichProgressBar,
    RichProgressBarTheme,
)

import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--conf_dir",
    default="configs/lrs2_SwiftNet_6.yml",
    help="Path to the YAML configuration file.",
)

parser.add_argument(
    "--precision",
    type=str,
    default="32-true",
    choices=["32-true", "bf16-mixed", "16-mixed"],
    help=(
        "Training precision: "
        "'32-true', 'bf16-mixed', or '16-mixed'."
    ),
)

parser.add_argument(
    "--tf32",
    action="store_true",
    help="Enable TF32 for FP32 matrix multiplications and CUDA convolutions.",
)

parser.add_argument(
    "--torch_compile",
    action="store_true",
    help="Compile the LightningModule using torch.compile().",
)

parser.add_argument(
    "--compile_mode",
    type=str,
    default="default",
    choices=["default", "reduce-overhead", "max-autotune"],
    help="Mode used by torch.compile().",
)

parser.add_argument(
    "--wandb_run_id",
    type=str,
    default=None,
    help=(
        "ID of an existing W&B run. If specified, the run is "
        "resumed with resume='must'. Normally not needed for new training."
    ),
)


def get_unique_experiment_name(base_name, experiments_root):
    """
    Return a unique experiment name.

    Example:
        LRS2_SwiftNet_6
        LRS2_SwiftNet_6-1
        LRS2_SwiftNet_6-2
    """
    os.makedirs(experiments_root, exist_ok=True)

    existing_names = set(os.listdir(experiments_root))

    if base_name not in existing_names:
        return base_name

    pattern = re.compile(rf"^{re.escape(base_name)}-(\d+)$")
    suffixes = []

    for existing_name in existing_names:
        match = pattern.match(existing_name)

        if match:
            suffixes.append(int(match.group(1)))

    next_suffix = max(suffixes, default=0) + 1

    while f"{base_name}-{next_suffix}" in existing_names:
        next_suffix += 1

    return f"{base_name}-{next_suffix}"


def configure_numerical_precision(tf32_enabled):
    """
    Configure the behavior of FP32 operations on CUDA.

    If TF32 is enabled:
        - FP32 matrix multiplications can use TF32 Tensor Cores;
        - FP32 cuDNN convolutions can use TF32.

    Mixed precision is configured separately in the Trainer.
    """
    if not torch.cuda.is_available():
        print_only("CUDA unavailable: ignoring TF32 configuration.")
        return

    if tf32_enabled:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print_only("TF32 enabled for FP32 matrix multiplications and cuDNN convolutions.")
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        print_only("TF32 disabled.")


def build_wandb_logger(
    experiment_name,
    logger_dir,
    config,
    wandb_run_id=None,
):
    """
    Build the W&B logger.

    If wandb_run_id is specified, obligatorily continue that run.
    """
    logger_kwargs = {
        "name": experiment_name,
        "project": "SwiftNet",
        "save_dir": logger_dir,
        "log_model": False,
    }

    if wandb_run_id is not None:
        logger_kwargs["id"] = wandb_run_id
        logger_kwargs["resume"] = "must"

        print_only(
            f"Resuming W&B run with ID: {wandb_run_id}"
        )

    wandb_logger = WandbLogger(**logger_kwargs)
    wandb_logger.log_hyperparams(config)

    return wandb_logger


def main(
    config,
    precision,
    tf32_enabled,
    torch_compile_enabled,
    compile_mode,
    wandb_run_id,
):
    configure_numerical_precision(tf32_enabled)

    # ------------------------------------------------------------------
    # Unique experiment name
    # ------------------------------------------------------------------
    experiments_root = os.path.join(
        os.getcwd(),
        "Experiments",
    )

    base_experiment_name = config["exp"]["exp_name"]

    experiment_name = get_unique_experiment_name(
        base_name=base_experiment_name,
        experiments_root=experiments_root,
    )

    config["exp"]["base_exp_name"] = base_experiment_name
    config["exp"]["exp_name"] = experiment_name

    print_only(
        f"Requested experiment name: {base_experiment_name}"
    )
    print_only(
        f"Used experiment name: {experiment_name}"
    )

    # Also save runtime options in the configuration.
    config.setdefault("runtime", {})

    config["runtime"]["precision"] = precision
    config["runtime"]["tf32"] = tf32_enabled
    config["runtime"]["torch_compile"] = torch_compile_enabled
    config["runtime"]["compile_mode"] = compile_mode
    config["runtime"]["wandb_run_id"] = wandb_run_id

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    print_only(
        "Instantiating datamodule <{}>".format(
            config["datamodule"]["data_name"]
        )
    )

    datamodule: object = getattr(
        look2hear.datas,
        config["datamodule"]["data_name"],
    )(
        **config["datamodule"]["data_config"]
    )

    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader

    # ------------------------------------------------------------------
    # Audio model
    # ------------------------------------------------------------------
    print_only(
        "Instantiating AudioNet <{}>".format(
            config["audionet"]["audionet_name"]
        )
    )

    model = getattr(
        look2hear.models,
        config["audionet"]["audionet_name"],
    )(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )

    # ------------------------------------------------------------------
    # Video model
    # ------------------------------------------------------------------
    print_only(
        "Instantiating VideoNet <{}>".format(
            config["videonet"]["videonet_name"]
        )
    )

    video_model = getattr(
        look2hear.videomodels,
        config["videonet"]["videonet_name"],
    )(
        **config["videonet"]["videonet_config"],
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    print_only(
        "Instantiating Optimizer <{}>".format(
            config["optimizer"]["optim_name"]
        )
    )

    optimizer = make_optimizer(
        model.parameters(),
        **config["optimizer"],
    )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    scheduler = None

    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(
                config["scheduler"]["sche_name"]
            )
        )

        scheduler = getattr(
            torch.optim.lr_scheduler,
            config["scheduler"]["sche_name"],
        )(
            optimizer=optimizer,
            **config["scheduler"]["sche_config"],
        )

    # ------------------------------------------------------------------
    # Experiment directories
    # ------------------------------------------------------------------
    exp_dir = os.path.join(
        experiments_root,
        experiment_name,
    )

    checkpoint_dir = os.path.join(
        exp_dir,
        "checkpoints",
    )

    wandb_dir = os.path.join(
        exp_dir,
        "wandb",
    )

    # experiment_name è già stato reso univoco, quindi questa cartella
    # non dovrebbe esistere.
    os.makedirs(exp_dir, exist_ok=False)
    os.makedirs(checkpoint_dir, exist_ok=False)
    os.makedirs(wandb_dir, exist_ok=False)

    config.setdefault("main_args", {})
    config["main_args"]["exp_dir"] = exp_dir
    config["main_args"]["checkpoint_dir"] = checkpoint_dir
    config["main_args"]["wandb_dir"] = wandb_dir

    print_only(f"Experiment directory: {exp_dir}")
    print_only(f"Checkpoint directory: {checkpoint_dir}")
    print_only(f"W&B directory: {wandb_dir}")

    conf_path = os.path.join(exp_dir, "conf.yml")

    with open(conf_path, "w") as outfile:
        yaml.safe_dump(
            config,
            outfile,
            sort_keys=False,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    print_only(
        "Instantiating Loss, Train <{}>, Val <{}>".format(
            config["loss"]["train"]["sdr_type"],
            config["loss"]["val"]["sdr_type"],
        )
    )

    loss_func = {
        "train": getattr(
            look2hear.losses,
            config["loss"]["train"]["loss_func"],
        )(
            getattr(
                look2hear.losses,
                config["loss"]["train"]["sdr_type"],
            ),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(
            look2hear.losses,
            config["loss"]["val"]["loss_func"],
        )(
            getattr(
                look2hear.losses,
                config["loss"]["val"]["sdr_type"],
            ),
            **config["loss"]["val"]["config"],
        ),
    }

    # ------------------------------------------------------------------
    # Lightning System
    # ------------------------------------------------------------------
    print_only(
        "Instantiating System <{}>".format(
            config["training"]["system"]
        )
    )

    system = getattr(
        look2hear.system,
        config["training"]["system"],
    )(
        audio_model=model,
        video_model=video_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    # ------------------------------------------------------------------
    # Torch compile
    # ------------------------------------------------------------------
    if torch_compile_enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "torch.compile non è disponibile in questa versione di PyTorch."
            )

        print_only(
            f"Compiling model with torch.compile(mode='{compile_mode}')."
        )

        torch._dynamo.config.suppress_errors = True

        system = torch.compile(
            system,
            mode=compile_mode,
            fullgraph=False,
        )
    else:
        print_only("torch.compile disabled.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    print_only("Instantiating ModelCheckpoint")

    callbacks = []

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:03d}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
        auto_insert_metric_name=False,
    )

    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")

        callbacks.append(
            EarlyStopping(
                **config["training"]["early_stop"]
            )
        )

    # callbacks.append(
    #     MyRichProgressBar(
    #         theme=RichProgressBarTheme()
    #     )
    # )

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------
    gpus = (
        config["training"]["gpus"]
        if torch.cuda.is_available()
        else 1
    )

    distributed_backend = (
        "gpu"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    wandb_logger = build_wandb_logger(
        experiment_name=experiment_name,
        logger_dir=wandb_dir,
        config=config,
        wandb_run_id=wandb_run_id,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(
            find_unused_parameters=True
        ),
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
        logger=wandb_logger,
        sync_batchnorm=True,
        num_sanity_val_steps=0,

        # Number of batches over which to accumulate gradients before
        # running optimizer.step(). Changes the effective batch size.
        accumulate_grad_batches=config["training"][
            "accumulate_grad_batches"
        ],

        # 32-true, bf16-mixed, or 16-mixed.
        precision=precision,
    )

    trainer.fit(system)

    print_only("Finished Training")

    # ------------------------------------------------------------------
    # Save best checkpoints information
    # ------------------------------------------------------------------
    best_k = {
        path: score.item()
        for path, score in checkpoint.best_k_models.items()
    }

    with open(
        os.path.join(exp_dir, "best_k_models.json"),
        "w",
    ) as file:
        json.dump(
            best_k,
            file,
            indent=2,
        )

    if not checkpoint.best_model_path:
        raise RuntimeError(
            "No best checkpoint available. "
            "Check the ModelCheckpoint monitor."
        )

    # ------------------------------------------------------------------
    # Export best model
    # ------------------------------------------------------------------
    state_dict = torch.load(
        checkpoint.best_model_path,
        map_location="cpu",
        weights_only=False,
    )

    system.load_state_dict(
        state_dict=state_dict["state_dict"]
    )

    system.cpu()

    to_save = system.audio_model.serialize()

    torch.save(
        to_save,
        os.path.join(exp_dir, "best_model.pth"),
    )


if __name__ == "__main__":
    import yaml

    from look2hear.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    # First read only the known arguments so we can find the YAML file
    # before dynamically adding its options.
    preliminary_args, _ = parser.parse_known_args()

    with open(preliminary_args.conf_dir) as file:
        def_conf = yaml.safe_load(file)

    parser = prepare_parser_from_dict(
        def_conf,
        parser=parser,
    )

    arg_dic, plain_args = parse_args_as_dict(
        parser,
        return_plain_args=True,
    )

    main(
        config=arg_dic,
        precision=plain_args.precision,
        tf32_enabled=plain_args.tf32,
        torch_compile_enabled=plain_args.torch_compile,
        compile_mode=plain_args.compile_mode,
        wandb_run_id=plain_args.wandb_run_id,
    )
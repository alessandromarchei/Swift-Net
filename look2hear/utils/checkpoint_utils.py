"""Checkpoint helpers: monitor-name migration, visual-mode key handling, architecture checks and weight loading.

Nothing here loads weights silently: every skipped or missing key is reported, and a
architecture mismatch raises unless the caller asks for partial loading.
"""

import copy

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

LEGACY_MONITOR = "val_loss/dataloader_idx_0"
MONITOR = "val_loss"


def migrate_checkpoint(src, dst, drop_video_model=False):
    """Write a copy of a Lightning checkpoint with the monitor renamed and optionally without the video model.

    * Callback states (ModelCheckpoint, EarlyStopping) are keyed by their monitor name. They are renamed so
      that resuming keeps the saved top-k history and early-stopping counter instead of resetting them.
    * Optimizer and LR-scheduler states are copied unchanged.
    * ``drop_video_model=True`` removes ``video_model.*`` weights, needed when resuming an online
      checkpoint in cached-visual mode (the cached mode has no video model).

    Returns a dict describing what changed.
    """
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    renamed = []
    new_callbacks = {}
    for key, state in ckpt.get("callbacks", {}).items():
        new_key = key.replace(LEGACY_MONITOR, MONITOR)
        if new_key != key:
            renamed.append((key, new_key))
        new_callbacks[new_key] = state
    ckpt["callbacks"] = new_callbacks

    dropped = []
    if drop_video_model:
        dropped = [k for k in ckpt["state_dict"] if k.startswith("video_model.")]
        ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("video_model.")}

    torch.save(ckpt, dst)
    return {"renamed_callback_states": renamed, "dropped_video_keys": len(dropped)}


def checkpoint_architecture(ckpt):
    """Architecture stored in a checkpoint. Old checkpoints (before this metadata) are the original GRNN model."""
    if "architecture" in ckpt:
        return ckpt["architecture"]
    return {"frequency_mixer": {"type": "grnn"}}


def check_architecture(ckpt, expected, allow_mismatch=False):
    """Raise if the checkpoint was trained with a different frequency modeling than ``expected``.

    Only the fields present in the checkpoint metadata are compared.
    """
    found = checkpoint_architecture(ckpt)
    mismatches = []
    for field in ("frequency_mixer", "ftgs_repeats", "ftgs_shared"):
        if field not in found or field not in expected:
            continue
        found_val = found[field]
        exp_val = expected[field]
        if field == "frequency_mixer":
            found_val = {"type": found_val.get("type", "grnn")}
            exp_val = {"type": exp_val.get("type", "grnn")}
        if found_val != exp_val:
            mismatches.append(f"{field}: checkpoint={found_val} config={exp_val}")
    if mismatches and not allow_mismatch:
        raise ValueError(
            "Checkpoint architecture differs from the configuration:\n  "
            + "\n  ".join(mismatches)
            + "\nResume with the same architecture, or initialize with --init_ckpt and --init_mode partial."
        )
    return mismatches


def load_audio_weights(audio_model, state_dict, mode="strict"):
    """Load audio-model weights from a Lightning or plain state dict.

    mode="strict":  every key must match (exact same architecture).
    mode="partial": load only keys present in the model with identical shapes; everything else is
                    reported. Used for explicit transfer learning between architectures.
    """
    if mode not in ("strict", "partial"):
        raise ValueError("mode must be 'strict' or 'partial'")

    prefix = "audio_model."
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    if mode == "strict":
        audio_model.load_state_dict(state_dict, strict=True)
        return {"loaded": len(state_dict), "skipped": [], "missing": []}

    own = audio_model.state_dict()
    to_load = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(k for k in state_dict if k not in to_load)
    missing = sorted(k for k in own if k not in to_load)
    audio_model.load_state_dict(to_load, strict=False)
    return {"loaded": len(to_load), "skipped": skipped, "missing": missing}


def upgrade_legacy_monitor(config):
    """Replace the legacy early-stopping monitor name in a loaded config. Prints a warning when it changes."""
    es = config.get("training", {}).get("early_stop")
    if es and es.get("monitor") == LEGACY_MONITOR:
        es["monitor"] = MONITOR
        print(f"Warning: early_stop.monitor '{LEGACY_MONITOR}' is renamed to '{MONITOR}'.")
    return config


# ReduceLROnPlateau settings taken from the config on resume. The progress (best, num_bad_epochs,
# cooldown_counter, last_epoch) stays the checkpoint's.
_PLATEAU_SETTINGS = ("factor", "patience", "cooldown", "threshold", "threshold_mode", "eps", "min_lrs")


class ApplyConfigOnResume(pl.Callback):
    """Make the config, not the checkpoint, decide ReduceLROnPlateau and EarlyStopping settings on resume.

    Lightning restores whole objects: ``ReduceLROnPlateau.load_state_dict`` overwrites patience, factor, ...
    and ``EarlyStopping.load_state_dict`` overwrites patience with the values of the original run. This
    callback records the values built from the config and puts them back in ``on_train_start``, which
    runs after the checkpoint restore. Counters are kept: if ``num_bad_epochs`` (or ``wait_count``) is
    already above the new patience, the next bad epoch reduces the lr (or stops training).
    """

    def __init__(self, scheduler=None, early_stopping=None):
        self.plateau = None
        if isinstance(scheduler, ReduceLROnPlateau):
            self.plateau = {k: copy.deepcopy(getattr(scheduler, k)) for k in _PLATEAU_SETTINGS}
            # TwoStagePlateauLR: ``patience`` above is the initial-stage one; the stage is re-derived below
            for k in ("initial_patience", "patience_after_reduce"):
                if hasattr(scheduler, k):
                    self.plateau[k] = getattr(scheduler, k)
            self.plateau_mode = scheduler.mode
        self.es_patience = early_stopping.patience if early_stopping is not None else None

    def on_train_start(self, trainer, pl_module):
        changes = []
        if self.plateau is not None:
            for cfg in trainer.lr_scheduler_configs:
                sched = cfg.scheduler
                if not isinstance(sched, ReduceLROnPlateau):
                    continue
                if sched.mode != self.plateau_mode:
                    raise ValueError(
                        f"ReduceLROnPlateau mode changed from {sched.mode!r} (checkpoint) to "
                        f"{self.plateau_mode!r} (config): the restored best value would be meaningless."
                    )
                for key, value in self.plateau.items():
                    if getattr(sched, key, None) != value:
                        changes.append(f"scheduler.{key}: {getattr(sched, key, None)} -> {value}")
                        setattr(sched, key, copy.deepcopy(value))
                if hasattr(sched, "update_patience"):
                    sched.update_patience()
                    changes.append(f"(two-stage plateau: lr reduced={sched.lr_was_reduced()}, active patience={sched.patience})")
                sched._init_is_better(mode=sched.mode, threshold=sched.threshold, threshold_mode=sched.threshold_mode)
                changes.append(f"(scheduler num_bad_epochs={sched.num_bad_epochs}, best={float(sched.best):.4f})")
        if self.es_patience is not None:
            for cb in trainer.callbacks:
                if isinstance(cb, EarlyStopping) and cb.patience != self.es_patience:
                    changes.append(f"early_stop.patience: {cb.patience} -> {self.es_patience}")
                    cb.patience = self.es_patience
        if trainer.is_global_zero:
            print("Resume: settings from the config applied over the checkpoint: " + "; ".join(changes or ["none"]))

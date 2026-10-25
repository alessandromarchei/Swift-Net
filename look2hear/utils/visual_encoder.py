"""Training switch for the visual encoder (FRCNNVideoModel): frozen (default) or fine-tuned with the separator.

Config, in the ``videonet`` block:
    freeze_visual_encoder: true   # default when absent. false: the encoder is updated with the separator.
    visual_lr: null               # encoder learning rate when trainable. null: the optimizer lr.

A trainable encoder needs online visual input (datamodule.visual_features_path: null): precomputed
embeddings have no encoder to update.

The exported ``best_model.pth`` is the same file in both cases. When the encoder is trainable, its
updated weights are stored under the ``visual_encoder`` key of that file, so the separator and the
encoder stay one artifact. Frozen runs export exactly the file they always did.
"""

# Keys of the videonet config that describe the encoder constructor. "pretrain" is dropped on export:
# the embedded state dict already contains the fine-tuned weights.
_EXPORT_EXCLUDED_KEYS = ("pretrain",)


def visual_encoder_trainable(videonet_cfg):
    """True when the encoder must be updated. Missing key means frozen (previous behaviour)."""
    return not bool(videonet_cfg.get("freeze_visual_encoder", True))


def validate_visual_training(trainable, visual_mode):
    """Raise if the encoder is asked to train without an encoder in the forward pass."""
    if trainable and visual_mode != "online":
        raise ValueError(
            "videonet.freeze_visual_encoder: false requires online visual input "
            "(datamodule.data_config.visual_features_path: null). Precomputed embeddings have no encoder to update."
        )


def build_optimizer_params(audio_model, video_model, trainable, visual_lr=None):
    """Parameters for make_optimizer.

    Frozen: the plain separator parameters, exactly as before.
    Trainable: two groups. The separator uses the optimizer lr. The encoder uses ``visual_lr`` if given,
    otherwise it inherits the optimizer lr.
    """
    if not trainable:
        return audio_model.parameters()

    visual_group = {"params": list(video_model.parameters())}
    if visual_lr is not None:
        visual_group["lr"] = visual_lr
    return [{"params": list(audio_model.parameters())}, visual_group]


def check_resume_compatibility(checkpoint, trainable):
    """Refuse to resume across a frozen/trainable change: the saved optimizer state has a different layout.

    Checkpoints written before this flag existed were frozen.
    """
    saved = bool(checkpoint.get("visual_encoder_trainable", False))
    if saved != trainable:
        raise ValueError(
            f"Checkpoint was trained with visual encoder trainable={saved}, but the configuration says "
            f"trainable={trainable}. Resuming across this change would mismatch the optimizer state. "
            "Set videonet.freeze_visual_encoder to match the checkpoint, or start a new run with --init_ckpt."
        )


def export_payload(system, videonet_cfg):
    """Serialized model written to best_model.pth. ``system`` is the non-compiled Lightning module."""
    payload = system.audio_model.serialize()
    if system.visual_trainable:
        payload["visual_encoder"] = {
            "videonet_name": videonet_cfg["videonet_name"],
            "videonet_config": {
                k: v for k, v in videonet_cfg["videonet_config"].items() if k not in _EXPORT_EXCLUDED_KEYS
            },
            "state_dict": system.video_model.state_dict(),
        }
    return payload

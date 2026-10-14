"""Tests for the visual-encoder training switch (videonet.freeze_visual_encoder).

Run from the repository root:
    python -m unittest tests.test_visual_encoder_training -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn

from look2hear.system import AudioVisualLightningModule, make_optimizer
from look2hear.utils.visual_encoder import (
    build_optimizer_params,
    check_resume_compatibility,
    export_payload,
    validate_visual_training,
    visual_encoder_trainable,
)


class Video(nn.Module):
    """Tiny stand-in for FRCNNVideoModel: [B, 1, T, H, W] -> [B, D, T]."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=(3, 4))


class Audio(nn.Module):
    """Tiny stand-in for the separator. Its output depends on the visual embedding."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, wav, emb):
        return (self.gain * wav + emb.mean(dim=(1, 2))[:, None]).unsqueeze(1)

    def serialize(self):
        return {"model_name": "Audio", "state_dict": self.state_dict(), "model_args": {}}


def _system(visual_trainable, video=None):
    return AudioVisualLightningModule(
        audio_model=Audio(),
        video_model=Video() if video is None else video,
        loss_func={"train": None, "val": None},
        visual_mode="online",
        visual_trainable=visual_trainable,
    )


def _one_step(system, lr=0.1):
    """One optimizer step on the system's parameters, exactly as the training script builds them."""
    trainable = system.visual_trainable
    params = build_optimizer_params(system.audio_model, system.video_model, trainable)
    opt = make_optimizer(params, optim_name="SGD", lr=lr)
    wav = torch.randn(2, 1600)
    mouth = torch.randn(2, 5, 8, 8)
    est = system(wav, mouth)
    est.pow(2).mean().backward()
    opt.step()
    return opt


class FlagParsingTests(unittest.TestCase):
    def test_missing_flag_means_frozen(self):
        # Configs written before the flag existed must keep training the encoder frozen.
        self.assertFalse(visual_encoder_trainable({}))
        self.assertFalse(visual_encoder_trainable({"videonet_name": "FRCNNVideoModel"}))

    def test_explicit_values(self):
        self.assertFalse(visual_encoder_trainable({"freeze_visual_encoder": True}))
        self.assertTrue(visual_encoder_trainable({"freeze_visual_encoder": False}))

    def test_trainable_is_rejected_with_cached_embeddings(self):
        with self.assertRaises(ValueError):
            validate_visual_training(True, "cached")
        validate_visual_training(False, "cached")
        validate_visual_training(True, "online")

    def test_system_rejects_trainable_without_online_mode(self):
        with self.assertRaises(ValueError):
            AudioVisualLightningModule(
                audio_model=Audio(),
                video_model=None,
                loss_func={"train": None, "val": None},
                visual_mode="cached",
                visual_trainable=True,
            )


class TrainingBehaviourTests(unittest.TestCase):
    def test_frozen_encoder_gets_no_gradient_and_is_not_updated(self):
        system = _system(visual_trainable=False)
        before = {k: v.clone() for k, v in system.video_model.state_dict().items()}
        opt = _one_step(system)

        self.assertTrue(all(not p.requires_grad for p in system.video_model.parameters()))
        self.assertTrue(all(p.grad is None for p in system.video_model.parameters()))
        for k, v in system.video_model.state_dict().items():
            self.assertTrue(torch.equal(before[k], v), k)
        # Only the separator is in the optimizer, as before the flag existed.
        self.assertEqual(len(opt.param_groups), 1)

    def test_trainable_encoder_gets_gradient_and_is_updated_with_separator(self):
        system = _system(visual_trainable=True)
        before = {k: v.clone() for k, v in system.video_model.state_dict().items()}
        audio_before = system.audio_model.gain.detach().clone()
        opt = _one_step(system)

        self.assertTrue(all(p.requires_grad for p in system.video_model.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.video_model.parameters()))
        changed = [k for k, v in system.video_model.state_dict().items() if not torch.equal(before[k], v)]
        self.assertTrue(changed, "encoder weights did not change")
        self.assertFalse(torch.equal(audio_before, system.audio_model.gain))
        # Separator and encoder are two parameter groups of the same optimizer.
        self.assertEqual(len(opt.param_groups), 2)

    def test_visual_lr_applies_only_to_encoder_group(self):
        system = _system(visual_trainable=True)
        params = build_optimizer_params(system.audio_model, system.video_model, True, visual_lr=1e-4)
        opt = make_optimizer(params, optim_name="SGD", lr=1e-2)
        self.assertEqual(opt.param_groups[0]["lr"], 1e-2)
        self.assertEqual(opt.param_groups[1]["lr"], 1e-4)

    def test_visual_lr_none_inherits_optimizer_lr(self):
        system = _system(visual_trainable=True)
        params = build_optimizer_params(system.audio_model, system.video_model, True, visual_lr=None)
        opt = make_optimizer(params, optim_name="SGD", lr=1e-2)
        self.assertEqual(opt.param_groups[1]["lr"], 1e-2)

    def test_frozen_optimizer_params_are_unchanged(self):
        audio = Audio()
        self.assertIs(type(build_optimizer_params(audio, Video(), False)), type(audio.parameters()))


class ExportTests(unittest.TestCase):
    def test_frozen_export_is_the_separator_only(self):
        system = _system(visual_trainable=False)
        cfg = {"videonet_name": "Video", "videonet_config": {"pretrain": "x.tar"}}
        payload = export_payload(system, cfg)
        self.assertNotIn("visual_encoder", payload)
        self.assertEqual(set(payload.keys()), {"model_name", "state_dict", "model_args"})

    def test_trainable_export_carries_updated_encoder_in_same_file(self):
        system = _system(visual_trainable=True)
        _one_step(system)  # make the encoder differ from its initialization
        cfg = {"videonet_name": "Video", "videonet_config": {"pretrain": "x.tar", "width": 4}}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best_model.pth")
            torch.save(export_payload(system, cfg), path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(loaded["model_name"], "Audio")
        self.assertIn("visual_encoder", loaded)
        enc = loaded["visual_encoder"]
        self.assertEqual(enc["videonet_name"], "Video")
        self.assertEqual(enc["videonet_config"], {"width": 4})  # pretrain path not exported
        for k, v in system.video_model.state_dict().items():
            self.assertTrue(torch.equal(enc["state_dict"][k], v), k)

    def test_exported_encoder_reproduces_training_output(self):
        system = _system(visual_trainable=True)
        _one_step(system)
        payload = export_payload(system, {"videonet_name": "Video", "videonet_config": {}})

        fresh = Video()
        fresh.load_state_dict(payload["visual_encoder"]["state_dict"], strict=True)
        mouth = torch.randn(2, 5, 8, 8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(system.video_model(mouth.unsqueeze(1)), fresh(mouth.unsqueeze(1))))


class ResumeTests(unittest.TestCase):
    def test_same_mode_resumes(self):
        check_resume_compatibility({"visual_encoder_trainable": True}, True)
        check_resume_compatibility({"visual_encoder_trainable": False}, False)

    def test_old_checkpoint_without_flag_counts_as_frozen(self):
        check_resume_compatibility({}, False)
        with self.assertRaises(ValueError):
            check_resume_compatibility({}, True)

    def test_switching_mode_on_resume_is_refused(self):
        with self.assertRaises(ValueError):
            check_resume_compatibility({"visual_encoder_trainable": True}, False)
        with self.assertRaises(ValueError):
            check_resume_compatibility({"visual_encoder_trainable": False}, True)

    def test_checkpoint_records_the_flag(self):
        system = _system(visual_trainable=True)
        ckpt = system.on_save_checkpoint({})
        self.assertTrue(ckpt["visual_encoder_trainable"])


if __name__ == "__main__":
    unittest.main()

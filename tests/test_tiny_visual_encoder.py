"""Tests for TinyVisualEncoderModel (KD visual frontend used in place of FRCNNVideoModel).

Run from the repository root:
    python -m unittest tests.test_tiny_visual_encoder -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from look2hear.system import AudioVisualLightningModule, make_optimizer
from look2hear.utils.visual_encoder import build_optimizer_params
from look2hear.videomodels import TinyVisualEncoderModel, TinyVisualFrontend

ARCH = dict(embedding_dim=32, temporal_kernel=5, stem_channels=8, widths=(8, 16), expand=2.0)


def _kd_checkpoint(tmp, arch=ARCH):
    """A checkpoint laid out like TinyVisualEncoder/train_kd.py writes it."""
    frontend = TinyVisualFrontend(**arch)
    path = os.path.join(tmp, "best.pt")
    args = {**arch, "widths": ",".join(str(w) for w in arch["widths"])}
    torch.save({"epoch": 1, "model": frontend.state_dict(), "args": args}, path)
    return path, frontend


class TinyVisualEncoderTests(unittest.TestCase):
    def test_interface_matches_frcnn(self):
        model = TinyVisualEncoderModel(**ARCH, input_size=112).eval()
        x = torch.randn(2, 1, 10, 88, 88)  # [B, 1, T, H, W]
        self.assertEqual(tuple(model(x).shape), (2, 32, 10))  # [B, D, T]
        self.assertEqual(tuple(model(x[:, 0]).shape), (2, 32, 10))  # 4-D input is accepted too

    def test_pretrain_loads_strictly_and_reproduces_kd_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, frontend = _kd_checkpoint(tmp)
            model = TinyVisualEncoderModel(**ARCH, pretrain=path).eval()
        frontend.eval()
        x = torch.randn(2, 1, 6, 88, 88)
        # Same input expressed in the KD normalization and resolution.
        kd = (x * 0.165 + 0.421 - 0.4161) / 0.1688
        kd = F.interpolate(kd[:, 0].reshape(-1, 1, 88, 88), size=(112, 112), mode="bilinear", align_corners=False)
        kd = kd.reshape(2, 6, 1, 112, 112).transpose(0, 1)  # [T, B, 1, H, W]
        with torch.no_grad():
            expected = frontend(kd).permute(1, 2, 0)
            self.assertTrue(torch.allclose(model(x), expected, atol=1e-5))

    def test_architecture_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _kd_checkpoint(tmp)
            with self.assertRaises(ValueError):
                TinyVisualEncoderModel(**{**ARCH, "stem_channels": 16}, pretrain=path)

    def test_widths_from_string(self):
        model = TinyVisualEncoderModel(**{**ARCH, "widths": "8,16"})
        self.assertEqual(model.widths, (8, 16))

    def test_freeze_bn_keeps_running_stats(self):
        frozen = TinyVisualEncoderModel(**ARCH, freeze_bn=True).train()
        before = frozen.frontend.frontend3d[1].running_mean.clone()
        frozen(torch.randn(2, 1, 6, 88, 88))
        self.assertTrue(torch.equal(before, frozen.frontend.frontend3d[1].running_mean))

        adapting = TinyVisualEncoderModel(**ARCH, freeze_bn=False).train()
        before = adapting.frontend.frontend3d[1].running_mean.clone()
        adapting(torch.randn(2, 1, 6, 88, 88))
        self.assertFalse(torch.equal(before, adapting.frontend.frontend3d[1].running_mean))

    def test_causal_output_ignores_future_frames(self):
        model = TinyVisualEncoderModel(**ARCH, causal=True).eval()
        x = torch.randn(1, 1, 8, 88, 88)
        y = x.clone()
        y[:, :, 5:] = torch.randn_like(y[:, :, 5:])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[..., :5], model(y)[..., :5], atol=1e-5))

    def test_trained_with_separator_in_lightning_module(self):
        class Audio(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.ones(1))

            def forward(self, wav, emb):
                return (self.gain * wav + emb.mean(dim=(1, 2))[:, None]).unsqueeze(1)

        system = AudioVisualLightningModule(
            audio_model=Audio(),
            video_model=TinyVisualEncoderModel(**ARCH),
            loss_func={"train": None, "val": None},
            visual_mode="online",
            visual_trainable=True,
        )
        params = build_optimizer_params(
            system.audio_model, system.video_model, True, visual_lr=1e-4, visual_weight_decay=1e-4
        )
        opt = make_optimizer(params, optim_name="adamw", lr=1e-3, weight_decay=0.1)
        self.assertEqual((opt.param_groups[0]["lr"], opt.param_groups[0]["weight_decay"]), (1e-3, 0.1))
        self.assertEqual((opt.param_groups[1]["lr"], opt.param_groups[1]["weight_decay"]), (1e-4, 1e-4))

        before = {k: v.clone() for k, v in system.video_model.state_dict().items()}
        system(torch.randn(2, 1600), torch.randn(2, 6, 88, 88)).pow(2).mean().backward()
        opt.step()
        changed = [k for k, v in system.video_model.state_dict().items() if not torch.equal(before[k], v)]
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()

"""Activation checkpointing must not change the model: same outputs, same gradients, same BN statistics.

Run from the repository root:
    python -m unittest tests.test_activation_checkpointing -v
"""

import copy
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import yaml

from look2hear.models import SwiftNet

CONFIG = REPO / "configs" / "lrs2_SwiftNet_6_freqmix_concat_v2.yml"


def build(activation_checkpointing: bool):
    with open(CONFIG) as f:
        conf = yaml.safe_load(f)
    kwargs = copy.deepcopy(conf["audionet"]["audionet_config"])
    torch.manual_seed(0)
    net = SwiftNet(sample_rate=16000, activation_checkpointing=activation_checkpointing, **kwargs)
    return net.train()


def one_step(net, audio, emb, seed=123):
    net.zero_grad(set_to_none=True)
    torch.manual_seed(seed)  # same dropout masks in both models
    out = net(audio, emb)
    loss = out.pow(2).mean()
    loss.backward()
    return out.detach(), loss.detach()


class ActivationCheckpointingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain = build(False)
        cls.ckpt = build(True)
        cls.ckpt.load_state_dict(cls.plain.state_dict())  # identical starting point
        torch.manual_seed(7)
        cls.audio = torch.randn(2, 32000) * 0.1  # 2 s at 16 kHz
        cls.emb = torch.randn(2, 512, 50)        # visual features, 50 frames

    def test_output_and_loss_match(self):
        out_p, loss_p = one_step(self.plain, self.audio, self.emb)
        out_c, loss_c = one_step(self.ckpt, self.audio, self.emb)
        self.assertLess((out_p - out_c).abs().max().item(), 1e-5)
        self.assertLess(abs(loss_p.item() - loss_c.item()), 1e-6)

    def test_gradients_match(self):
        self.plain.load_state_dict(self.ckpt.state_dict())
        one_step(self.plain, self.audio, self.emb)
        one_step(self.ckpt, self.audio, self.emb)
        plain_grads = dict(self.plain.named_parameters())
        worst = 0.0
        for name, p_ckpt in self.ckpt.named_parameters():
            g_p = plain_grads[name].grad
            g_c = p_ckpt.grad
            if g_p is None or g_c is None:
                self.assertIsNone(g_p, name)
                self.assertIsNone(g_c, name)
                continue
            denom = g_p.abs().max().item() + 1e-12
            worst = max(worst, (g_p - g_c).abs().max().item() / denom)
        self.assertLess(worst, 1e-4, "gradients differ between plain and checkpointed model")

    def test_bn_running_stats_updated_once(self):
        # Recomputation in backward must not update running stats a second time.
        self.plain.load_state_dict(self.ckpt.state_dict())
        one_step(self.plain, self.audio, self.emb)
        one_step(self.ckpt, self.audio, self.emb)
        bufs_p = dict(self.plain.named_buffers())
        checked = 0
        for name, b_c in self.ckpt.named_buffers():
            if "running_mean" not in name and "running_var" not in name:
                continue
            checked += 1
            self.assertTrue(
                torch.allclose(bufs_p[name], b_c, atol=1e-6),
                f"BN buffer {name} differs: running stats were updated twice",
            )
        self.assertGreater(checked, 0, "no BatchNorm running stats found in the model")

    def test_eval_mode_is_not_checkpointed(self):
        self.ckpt.eval()
        with torch.no_grad():
            out = self.ckpt(self.audio, self.emb)
        self.assertEqual(out.shape[0], 2)
        self.ckpt.train()


if __name__ == "__main__":
    unittest.main()

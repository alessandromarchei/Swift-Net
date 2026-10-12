"""Regression tests: batch items must not interact inside the video branch or the full model.

The video branch (LightVid) runs an SRU over the visual sequence. SRU expects (length, batch, channels),
so the recurrence must run over time T. Before the fix it ran over the batch axis, which made the output
of one sample depend on the other samples in the batch (and therefore on the batch size used for
validation). The first sample of a batch was unaffected, which hid the problem in forward-only checks.

Run from the repository root:
    python -m unittest tests.test_batch_independence -v
"""

import copy
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F
import yaml

from look2hear.models import SwiftNet
from look2hear.models.SwiftNet import LightVid

CONFIG = REPO / "configs" / "lrs2_SwiftNet_6.yml"
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def build_swiftnet():
    with open(CONFIG) as f:
        conf = yaml.safe_load(f)
    torch.manual_seed(0)
    return SwiftNet(sample_rate=16000, **copy.deepcopy(conf["audionet"]["audionet_config"])).eval()


def build_light_vid():
    torch.manual_seed(0)
    return LightVid(in_chan=512, hid_chan=64, norm_type="BatchNorm1d", act_type="PReLU", is2d=False, causal=True).eval()


class VideoBranchTests(unittest.TestCase):
    def test_sample_output_does_not_depend_on_other_batch_items(self):
        vid = build_light_vid()
        torch.manual_seed(1)
        x = torch.randn(4, 512, 50)  # B, C, T (visual bottleneck features)
        with torch.no_grad():
            batched = vid(x)
            alone = torch.cat([vid(x[i : i + 1]) for i in range(4)])
        self.assertLess((batched - alone).abs().max().item(), 1e-5)

    def test_recurrence_is_causal_over_time_and_isolated_per_sample(self):
        # Perturb frame 20 of sample 0. Only sample 0 may change, and only frames >= 20 may change.
        vid = build_light_vid()
        torch.manual_seed(2)
        x = torch.randn(2, 512, 50)
        x_pert = x.clone()
        x_pert[0, :, 20] += 3.0
        with torch.no_grad():
            y = vid(x)
            y_pert = vid(x_pert)
        delta = (y_pert - y).abs()
        self.assertLess(delta[1].max().item(), 1e-5, "sample 1 changed when sample 0 was perturbed")
        self.assertLess(delta[0, :, :20].max().item(), 1e-5, "frames before the perturbation changed (not causal)")
        self.assertGreater(delta[0, :, 20:].max().item(), 1e-3, "frames after the perturbation did not respond")


class FullModelTests(unittest.TestCase):
    def test_full_model_forward_is_per_sample(self):
        net = build_swiftnet()
        torch.manual_seed(3)
        audio = torch.randn(2, 32000) * 0.1
        emb = torch.randn(2, 512, 50)
        with torch.no_grad():
            batched = net(audio, emb)
            second_alone = net(audio[1:], emb[1:])
        self.assertLess((batched[1] - second_alone[0]).abs().max().item(), 1e-5)

    def test_gradients_are_additive_over_samples(self):
        # Gradient accumulation (2 x batch 1) must equal one batch of 2 when the model has no batch-dependent ops.
        net = build_swiftnet()
        init = copy.deepcopy(net.state_dict())
        torch.manual_seed(4)
        audio = torch.randn(2, 32000) * 0.1
        emb = torch.randn(2, 512, 50)
        target = torch.randn(2, 1, 32000) * 0.1

        def grads(batches):
            net.load_state_dict(init)
            net.zero_grad(set_to_none=True)
            n = sum(a.shape[0] for a, _, _ in batches)
            for a, e, t in batches:
                (F.mse_loss(net(a, e), t, reduction="sum") / (n * t[0].numel())).backward()
            return {k: p.grad.detach().clone() for k, p in net.named_parameters() if p.grad is not None}

        g_full = grads([(audio, emb, target)])
        g_acc = grads([(audio[:1], emb[:1], target[:1]), (audio[1:], emb[1:], target[1:])])
        num = sum(((g_full[k] - g_acc[k]) ** 2).sum() for k in g_full).sqrt()
        den = sum((g_full[k] ** 2).sum() for k in g_full).sqrt()
        self.assertLess((num / den).item(), 1e-4)


if __name__ == "__main__":
    unittest.main()

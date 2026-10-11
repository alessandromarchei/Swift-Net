"""Correctness tests for the optional convolutional frequency mixer and the SwiftNet integration.

Run from the repository root:
    python -m unittest tests.test_frequency_mixer -v
"""

import copy
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from look2hear.models.frequency_mixer import (
    DilatedFrequencyMixer,
    FrameGroupNorm,
    FrameLayerNorm,
    parse_frequency_mixer_config,
)
from look2hear.models import SwiftNet
from look2hear.models.SwiftNet import FTGS, Separator
from look2hear.utils.checkpoint_utils import check_architecture, load_audio_weights, migrate_checkpoint

CONFIG = REPO / "configs" / "lrs2_SwiftNet_6.yml"
CKPT = REPO / "Experiments" / "LRS2_SwiftNet_6" / "checkpoints" / "last.ckpt"
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def swiftnet_config(repeats=None, frequency_mixer="absent", short=True):
    """The real LRS2 SwiftNet-6 config, optionally with a different frequency mixer or repeat count."""
    with open(CONFIG) as f:
        conf = yaml.safe_load(f)
    ac = copy.deepcopy(conf["audionet"]["audionet_config"])
    if repeats is not None:
        ac["audio_params"]["repeats"] = repeats
    if frequency_mixer == "absent":
        ac["audio_params"].pop("frequency_mixer", None)
    elif frequency_mixer is not None:
        ac["audio_params"]["frequency_mixer"] = copy.deepcopy(frequency_mixer)
    return ac


def build_swiftnet(**kwargs):
    return SwiftNet(sample_rate=16000, **swiftnet_config(**kwargs))


CONV_SUM = {"type": "conv", "conv": {"kernel_sizes": [3, 3, 3], "dilations": [1, 2, 4], "branch_fusion": "sum"}}
CONV_CONCAT = {"type": "conv", "conv": {"kernel_sizes": [3, 3, 3], "dilations": [1, 2, 4], "branch_fusion": "concat", "activation": "gelu"}}


def reference_block(block, x):
    """Explicit re-implementation of DilatedFrequencyBlock.forward using only functional convs.

    Only valid for the default PReLU activation and pointwise/depthwise settings used in these tests.
    """
    y = block.norm(x)
    y = F.conv2d(y, block.expand.weight, block.expand.bias)
    y = F.prelu(y, block.act_in.weight)
    outs = [
        F.conv2d(y, b.weight, b.bias, dilation=b.dilation, padding=b.padding, groups=b.groups)
        for b in block.branches
    ]
    if block.branch_fusion == "concat":
        y = F.conv2d(torch.cat(outs, 1), block.fuse.weight, block.fuse.bias)
    else:
        y = sum(outs)
        if block.branch_fusion == "mean":
            y = y / len(outs)
    y = F.prelu(y, block.act_mid.weight)
    y = F.conv2d(y, block.project.weight, block.project.bias)
    return x + y if block.residual else y


class FrequencyMixerShapeTests(unittest.TestCase):
    def test_shape_preserved_over_sizes_and_channels(self):
        for C in (1, 8, 64):
            for F_ in (1, 5, 65):
                for T in (1, 17):
                    with self.subTest(C=C, F=F_, T=T):
                        m = DilatedFrequencyMixer(channels=C, dilations=[1, 2, 4])
                        x = torch.randn(2, C, T, F_)
                        self.assertEqual(m(x).shape, x.shape)

    def test_dilation_and_kernel_configurations(self):
        cases = [
            dict(dilations=[1, 2]),
            dict(dilations=[1, 2, 4]),
            dict(dilations=[1, 2, 4, 8]),
            dict(dilations=[1, 2, 4], kernel_sizes=[3, 5, 7]),
        ]
        for case in cases:
            with self.subTest(**case):
                m = DilatedFrequencyMixer(channels=16, **case)
                x = torch.randn(2, 16, 9, 33)
                self.assertEqual(m(x).shape, x.shape)
                self.assertTrue(torch.isfinite(m(x)).all())

    def test_branch_count_is_not_hardcoded(self):
        m = DilatedFrequencyMixer(channels=8, dilations=[1, 2, 4, 8, 16])
        self.assertEqual(len(m.blocks[0].branches), 5)

    def test_receptive_field_matches_formula(self):
        m = DilatedFrequencyMixer(channels=4, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4], residual=False)
        rf = [1 + (k - 1) * d for k, d in zip([3, 5, 7], [1, 2, 4])]
        self.assertEqual(rf, [3, 9, 25])
        for branch, k, d in zip(m.blocks[0].branches, [3, 5, 7], [1, 2, 4]):
            self.assertEqual(branch.kernel_size, (1, k))
            self.assertEqual(branch.dilation, (1, d))
            self.assertEqual(branch.padding, (0, d * (k - 1) // 2))


class FrequencyMixerNumericsTests(unittest.TestCase):
    def test_forward_matches_reference_for_all_fusions(self):
        torch.manual_seed(0)
        for fusion in ("sum", "mean", "concat"):
            for residual in (True, False):
                with self.subTest(fusion=fusion, residual=residual):
                    m = DilatedFrequencyMixer(
                        channels=8, dilations=[1, 2, 4], branch_fusion=fusion, residual=residual,
                        normalization="layernorm", activation="prelu",
                    ).eval()
                    x = torch.randn(2, 8, 5, 21)
                    with torch.no_grad():
                        torch.testing.assert_close(m(x), reference_block(m.blocks[0], x), rtol=1e-5, atol=1e-5)

    def test_residual_on_off(self):
        torch.manual_seed(1)
        on = DilatedFrequencyMixer(channels=8, dilations=[1, 2], residual=True).eval()
        off = DilatedFrequencyMixer(channels=8, dilations=[1, 2], residual=False).eval()
        off.load_state_dict(on.state_dict())
        x = torch.randn(1, 8, 4, 16)
        with torch.no_grad():
            torch.testing.assert_close(on(x), x + off(x), rtol=1e-5, atol=1e-5)

    def test_sum_and_mean_differ_by_branch_count(self):
        torch.manual_seed(2)
        s = DilatedFrequencyMixer(channels=8, dilations=[1, 2, 4], branch_fusion="sum", residual=False).eval()
        mn = DilatedFrequencyMixer(channels=8, dilations=[1, 2, 4], branch_fusion="mean", residual=False).eval()
        mn.load_state_dict(s.state_dict())
        x = torch.randn(1, 8, 3, 12)
        with torch.no_grad():
            # Same weights, different fusion: mean divides the branch sum by the branch count.
            self.assertFalse(torch.allclose(s(x), mn(x)))

    def test_no_nan_or_inf(self):
        torch.manual_seed(3)
        for fusion in ("sum", "mean", "concat"):
            for norm in ("layernorm", "groupnorm", "none"):
                m = DilatedFrequencyMixer(channels=16, dilations=[1, 2, 4], branch_fusion=fusion, normalization=norm)
                x = torch.randn(2, 16, 7, 65) * 10
                self.assertTrue(torch.isfinite(m(x)).all(), f"{fusion}/{norm}")

    def test_backward_gives_finite_gradients_for_every_parameter(self):
        torch.manual_seed(4)
        for fusion in ("sum", "concat"):
            with self.subTest(fusion=fusion):
                m = DilatedFrequencyMixer(channels=8, dilations=[1, 2, 4], branch_fusion=fusion)
                x = torch.randn(2, 8, 6, 20, requires_grad=True)
                m(x).pow(2).mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, p in m.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_optimizer_step_changes_parameters(self):
        torch.manual_seed(5)
        m = DilatedFrequencyMixer(channels=8, dilations=[1, 2])
        before = [p.detach().clone() for p in m.parameters()]
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        m(torch.randn(2, 8, 4, 16)).pow(2).mean().backward()
        opt.step()
        changed = [not torch.equal(b, p) for b, p in zip(before, m.parameters())]
        self.assertTrue(any(changed))
        self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))


class FrequencyOnlyTests(unittest.TestCase):
    """The mixer must never mix time frames, and must mix frequency bins."""

    def _check_temporal_independence(self, **kwargs):
        torch.manual_seed(10)
        m = DilatedFrequencyMixer(channels=8, **kwargs).eval()
        x = torch.randn(2, 8, 12, 33)
        t_cut = 6
        x_future = x.clone()
        x_future[:, :, t_cut:] += torch.randn_like(x_future[:, :, t_cut:]) * 5.0
        with torch.no_grad():
            y, y_future = m(x), m(x_future)
        # Earlier frames are bit-identical (no leakage from future frames).
        self.assertEqual((y[:, :, :t_cut] - y_future[:, :, :t_cut]).abs().max().item(), 0.0)
        # Sanity: future frames did change, so the test is not vacuous.
        self.assertGreater((y[:, :, t_cut:] - y_future[:, :, t_cut:]).abs().max().item(), 1e-3)

    def test_temporal_independence_all_normalizations(self):
        for norm in ("layernorm", "groupnorm", "none"):
            with self.subTest(normalization=norm):
                self._check_temporal_independence(dilations=[1, 2, 4], normalization=norm, branch_fusion="sum")

    def test_temporal_independence_concat_and_mean(self):
        for fusion in ("concat", "mean"):
            with self.subTest(fusion=fusion):
                self._check_temporal_independence(dilations=[1, 2, 4], branch_fusion=fusion, normalization="layernorm")

    def test_frequency_bins_are_mixed(self):
        # A change at one frequency bin must influence neighbouring bins.
        torch.manual_seed(11)
        m = DilatedFrequencyMixer(channels=8, dilations=[1, 2, 4], residual=False).eval()
        x = torch.randn(1, 8, 3, 33)
        x2 = x.clone()
        x2[..., 16] += 3.0
        with torch.no_grad():
            d = (m(x) - m(x2)).abs().sum(dim=(0, 1, 2))
        self.assertGreater(d[15].item() + d[17].item(), 0.0)
        self.assertEqual(d[0].item(), 0.0)  # far outside the receptive field of the change

    def test_normalizations_are_per_frame(self):
        torch.manual_seed(12)
        x = torch.randn(2, 8, 5, 17) * 3 + 2
        ln = FrameLayerNorm(8).eval()
        y = ln(x)
        # Each (batch, time, freq) position is normalized over channels only.
        torch.testing.assert_close(y.mean(dim=1), torch.zeros(2, 5, 17), atol=1e-5, rtol=0)
        gn = FrameGroupNorm(8, num_groups=4)
        # Frame statistics: changing other time frames must not change this frame's output.
        x2 = x.clone()
        x2[:, :, 3:] *= 10
        torch.testing.assert_close(gn(x)[:, :, :3], gn(x2)[:, :, :3], atol=1e-6, rtol=1e-6)


class FrequencyMixerConfigTests(unittest.TestCase):
    def test_default_and_absent_mean_grnn(self):
        self.assertEqual(parse_frequency_mixer_config(None), ("grnn", None))
        self.assertEqual(parse_frequency_mixer_config({"type": "grnn"}), ("grnn", None))

    def test_invalid_configs_raise(self):
        bad = [
            {"type": "lstm"},
            {"type": "grnn", "conv": {"dilations": [1]}},
            {"type": "conv", "conv": {"kernel_sizes": [3, 3], "dilations": [1]}},
            {"type": "conv", "conv": {"kernel_sizes": [4], "dilations": [1]}},
            {"type": "conv", "conv": {"dilations": [1], "branch_fusion": "max"}},
            {"type": "conv", "conv": {"dilations": [0]}},
            {"type": "conv", "conv": {"dilations": [1], "unknown_option": 1}},
            {"type": "conv", "conv": {"dilations": []}},
            {"type": "conv", "conv": {"dilations": [1], "activation": "tanh"}},
            {"type": "conv", "conv": {"dilations": [1], "normalization": "batchnorm"}},
            {"type": "conv", "conv": {"dilations": [1], "pointwise": False, "hidden_channels": 4}},
            {"typo": "grnn"},
        ]
        for cfg in bad:
            with self.subTest(cfg=cfg):
                with self.assertRaises(ValueError):
                    mtype, conv = parse_frequency_mixer_config(cfg)
                    if mtype == "conv":
                        DilatedFrequencyMixer(channels=8, **conv)

    def test_groupnorm_requires_divisible_channels(self):
        with self.assertRaises(ValueError):
            FrameGroupNorm(6, num_groups=4)


class SwiftNetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.audio = torch.randn(1, 8000)
        self.emb = torch.randn(1, 512, 25)

    def test_ftgs_frequency_module_type(self):
        grnn = build_swiftnet()
        conv = build_swiftnet(frequency_mixer=CONV_SUM)
        self.assertEqual(grnn.separator.audio_net.frequency_mixer_type, "grnn")
        self.assertEqual(conv.separator.audio_net.frequency_mixer_type, "conv")
        self.assertIsInstance(conv.separator.audio_net.process_f, DilatedFrequencyMixer)
        self.assertNotIsInstance(grnn.separator.audio_net.process_f, DilatedFrequencyMixer)

    def test_frequency_input_shape_is_B_C_T_F(self):
        # Hook the frequency module of the conv mixer and check the tensor shape it actually receives.
        net = build_swiftnet(frequency_mixer=CONV_SUM).eval()
        seen = {}
        net.separator.audio_net.process_f.register_forward_hook(
            lambda mod, inp, out: seen.update(inp=tuple(inp[0].shape), out=tuple(out.shape))
        )
        with torch.no_grad():
            net(self.audio, self.emb)
        self.assertEqual(seen["inp"], seen["out"])
        B, C, T, F_ = seen["inp"]
        self.assertEqual((B, C), (1, 64))
        self.assertEqual(F_, 65)  # n_freqs in layer_3 of the config

    def test_forward_with_conv_mixer_has_same_output_shape(self):
        for mixer in (None, CONV_SUM, CONV_CONCAT):
            with self.subTest(mixer=mixer):
                net = build_swiftnet(frequency_mixer=mixer if mixer is not None else "absent").eval()
                with torch.no_grad():
                    out = net(self.audio, self.emb)
                self.assertEqual(out.shape, (1, 1, 8000))
                self.assertTrue(torch.isfinite(out).all())

    def test_grnn_state_dict_keys_are_unchanged(self):
        absent = build_swiftnet()
        explicit = build_swiftnet(frequency_mixer={"type": "grnn"})
        self.assertEqual(sorted(absent.state_dict()), sorted(explicit.state_dict()))
        self.assertTrue(any(".process_f.rnn.rnn1." in k for k in absent.state_dict()))

    def test_conv_mixer_does_not_touch_temporal_grnn(self):
        grnn = build_swiftnet()
        conv = build_swiftnet(frequency_mixer=CONV_SUM)
        gk = {k: v.shape for k, v in grnn.state_dict().items() if ".process_t." in k}
        ck = {k: v.shape for k, v in conv.state_dict().items() if ".process_t." in k}
        self.assertEqual(gk, ck)
        self.assertTrue(gk)

    def test_shared_repeats_share_frequency_mixer(self):
        net = build_swiftnet(repeats=3, frequency_mixer=CONV_SUM)
        self.assertTrue(net.separator.audio_shared)
        self.assertIsInstance(net.separator.audio_net, FTGS)  # one instance used for all repeats

    def test_unshared_repeats_have_independent_frequency_mixers(self):
        ac = swiftnet_config(repeats=3, frequency_mixer=CONV_SUM)
        ac["audio_params"]["shared"] = False
        net = SwiftNet(sample_rate=16000, **ac)
        self.assertIsInstance(net.separator.audio_net, nn.ModuleList)
        ptrs = [blk.branches[0].weight.data_ptr() for blk in
                [ftgs.process_f.blocks[0] for ftgs in net.separator.audio_net]]
        self.assertEqual(len(set(ptrs)), len(ptrs))

    def test_repeats_in_config_are_respected(self):
        for r in (2, 3):
            net = build_swiftnet(repeats=r, frequency_mixer=CONV_SUM)
            self.assertEqual(net.architecture_summary()["ftgs_repeats"], r)

    def test_serialization_round_trip(self):
        torch.manual_seed(20)
        net = build_swiftnet(frequency_mixer=CONV_SUM).eval()
        buf = io.BytesIO()
        torch.save(net.state_dict(), buf)
        buf.seek(0)
        other = build_swiftnet(frequency_mixer=CONV_SUM).eval()
        other.load_state_dict(torch.load(buf, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(net(self.audio, self.emb), other(self.audio, self.emb))

    def test_optimizer_step_on_full_model(self):
        torch.manual_seed(21)
        net = build_swiftnet(frequency_mixer=CONV_CONCAT).train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        out = net(self.audio, self.emb)
        out.pow(2).mean().backward()
        opt.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in net.parameters()))

    def test_architecture_metadata_and_mismatch_check(self):
        conv = build_swiftnet(frequency_mixer=CONV_SUM)
        meta = conv.architecture_summary()
        self.assertEqual(meta["frequency_mixer"]["type"], "conv")
        # A legacy checkpoint (no metadata) is the original GRNN and must not be silently used for conv.
        with self.assertRaises(ValueError):
            check_architecture({"state_dict": {}}, meta)
        self.assertEqual(check_architecture({"state_dict": {}}, build_swiftnet().architecture_summary()), [])


@unittest.skipUnless(CKPT.exists(), "trained LRS2 checkpoint not available")
class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        self.audio_sd = {k[len("audio_model."):]: v for k, v in self.ckpt["state_dict"].items() if k.startswith("audio_model.")}

    def test_original_grnn_checkpoint_loads_strictly(self):
        net = build_swiftnet()
        load_audio_weights(net, self.audio_sd, mode="strict")

    def test_explicit_grnn_type_gives_identical_output(self):
        torch.manual_seed(30)
        a = build_swiftnet().eval()
        b = build_swiftnet(frequency_mixer={"type": "grnn"}).eval()
        load_audio_weights(a, self.audio_sd, mode="strict")
        load_audio_weights(b, self.audio_sd, mode="strict")
        audio, emb = torch.randn(1, 8000), torch.randn(1, 512, 25)
        with torch.no_grad():
            torch.testing.assert_close(a(audio, emb), b(audio, emb), rtol=0, atol=0)

    def test_grnn_checkpoint_into_conv_mixer_fails_strict(self):
        net = build_swiftnet(frequency_mixer=CONV_SUM)
        with self.assertRaises(RuntimeError):
            load_audio_weights(net, self.audio_sd, mode="strict")

    def test_partial_transfer_reports_what_was_skipped(self):
        net = build_swiftnet(frequency_mixer=CONV_SUM)
        report = load_audio_weights(net, self.audio_sd, mode="partial")
        self.assertGreater(report["loaded"], 0)
        self.assertTrue(any(".process_f.rnn." in k for k in report["skipped"]))
        self.assertTrue(any(".process_f.blocks." in k for k in report["missing"]))
        # Temporal GRNN weights were transferred exactly from the checkpoint.
        temporal = [k for k in net.state_dict() if ".process_t." in k]
        self.assertTrue(temporal)
        for k in temporal:
            self.assertTrue(torch.equal(net.state_dict()[k], self.audio_sd[k]), k)


if __name__ == "__main__":
    unittest.main()

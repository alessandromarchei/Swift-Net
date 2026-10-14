"""The XLA-native ops (look2hear.layers.xla_ops, dft_stft framing) against the PyTorch ops.

Run from the repository root (``PJRT_DEVICE=CPU`` is enough without a TPU):
    python -m unittest tests.test_xla_ops -v
"""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F

from look2hear.layers import dft_stft, xla_ops

try:
    import torch_xla
    import torch_xla.debug.metrics as met

    XLA_DEVICE = torch_xla.device()  # before any backward: autograd sizes its device queues once
except ImportError:
    torch_xla = None


def _rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def _compare(test, fn_ref, fn_xla, shape, exact):
    """fn_ref on CPU vs fn_xla on XLA: outputs and input gradients; no host fallback."""
    x = torch.randn(*shape)
    xc = x.clone().requires_grad_(True)
    yc = fn_ref(xc)
    gw = torch.randn(yc.shape)
    gc, = torch.autograd.grad((yc * gw).sum(), xc)

    met.clear_all()
    xx = x.to(XLA_DEVICE).requires_grad_(True)
    yx = fn_xla(xx)
    gx, = torch.autograd.grad((yx * gw.to(XLA_DEVICE)).sum(), xx)
    yx, gx = yx.cpu(), gx.cpu()
    test.assertEqual(met.executed_fallback_ops(), [])
    test.assertEqual(yx.shape, yc.shape)
    if exact:
        test.assertTrue(torch.equal(yx, yc.detach()))
        test.assertLess(_rel(gx, gc), 1e-6)  # the backward sums repeated indices
    else:
        test.assertLess(_rel(yx, yc), 1e-6)
        test.assertLess(_rel(gx, gc), 1e-6)


@unittest.skipIf(torch_xla is None, "needs torch_xla")
class TestXLAOps(unittest.TestCase):
    def test_interpolate_nearest(self):
        cases = [((2, 3, 50), 126), ((2, 3, 126), 50), ((2, 3, 63), 126), ((2, 3, 7), 7),
                 ((2, 4, 63, 33), (126, 65)), ((2, 4, 126, 65), (63, 33)), ((1, 2, 5, 9), (17, 4))]
        for shape, size in cases:
            with self.subTest(shape=shape, size=size):
                _compare(self, lambda x: F.interpolate(x, size=size, mode="nearest"),
                         lambda x: xla_ops.interpolate_nearest(x, size), shape, exact=True)

    def test_adaptive_avg_pool(self):
        cases = [((2, 3, 126, 65), (63, 33)), ((2, 3, 63, 33), (63, 33)), ((2, 3, 10, 7), (3, 5)),
                 ((2, 3, 126), (63,)), ((2, 3, 17), (4,))]
        for shape, size in cases:
            pool = F.adaptive_avg_pool2d if len(shape) == 4 else F.adaptive_avg_pool1d
            with self.subTest(shape=shape, size=size):
                _compare(self, lambda x: pool(x, size),
                         lambda x: xla_ops.adaptive_avg_pool(x, size), shape, exact=False)

    def test_unfold_kx1(self):
        for (k, s), shape in [((8, 1), (6, 4, 137, 1)), ((4, 2), (3, 5, 66, 3)), ((1, 1), (2, 2, 9, 1))]:
            with self.subTest(k=k, s=s):
                unfold = nn.Unfold((k, 1), stride=(s, 1))
                _compare(self, unfold, lambda x: xla_ops.unfold_kx1(x, k, s), shape, exact=True)

    def test_stft_frames(self):
        for n_fft, hop, L in [(256, 128, 32256), (256, 64, 16256), (512, 128, 24000)]:
            with self.subTest(n_fft=n_fft, hop=hop):
                _compare(self, lambda x: x.unfold(-1, n_fft, hop),
                         lambda x: dft_stft._frames(x, n_fft, hop), (3, L), exact=True)


@unittest.skipIf(torch_xla is None, "needs torch_xla")
class TestPITLossOnXLA(unittest.TestCase):
    def test_threshold_rule_matches_cpu(self):
        from look2hear.losses import PITLossWrapper, pairwise_neg_snr, pairwise_neg_sisdr

        # float64: at 60-80 dB SNR, est - target cancels and float32 rounding alone gives ~1e-3
        # relative noise on any device; the comparison targets the threshold rule, not rounding.
        g = torch.Generator().manual_seed(0)
        target = torch.randn(4, 1, 8000, generator=g, dtype=torch.float64)
        noise = torch.randn(4, 1, 8000, generator=g, dtype=torch.float64)
        # per-example SNR from very high (loss < -30, dropped) to low (kept)
        cases = {
            "mixed": torch.tensor([1e-3, 0.5, 1e-4, 0.1], dtype=torch.float64),
            "all kept": torch.tensor([0.5, 0.3, 0.2, 0.1], dtype=torch.float64),
            "all dropped": torch.tensor([1e-3, 1e-4, 1e-3, 1e-4], dtype=torch.float64),
        }
        for loss_func, threshold in ((pairwise_neg_snr, True), (pairwise_neg_sisdr, False)):
            wrapper = PITLossWrapper(loss_func, pit_from="pw_mtx", threshold_byloss=threshold)
            for name, scale in cases.items():
                with self.subTest(loss=loss_func.__name__ if hasattr(loss_func, "__name__") else str(loss_func), case=name):
                    est = (target + noise * scale.view(4, 1, 1)).requires_grad_(True)
                    ref = wrapper(est, target)
                    g_ref, = torch.autograd.grad(ref, est)
                    met.clear_all()
                    est_x = est.detach().to(XLA_DEVICE).requires_grad_(True)
                    got = wrapper(est_x, target.to(XLA_DEVICE))
                    g_got, = torch.autograd.grad(got, est_x)
                    self.assertLess(_rel(got, ref), 1e-6)  # a wrong threshold rule differs by O(1)
                    self.assertLess(_rel(g_got, g_ref), 1e-6)
                    self.assertEqual(met.executed_fallback_ops(), [])


if __name__ == "__main__":
    unittest.main()

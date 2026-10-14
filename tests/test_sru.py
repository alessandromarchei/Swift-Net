"""Equivalence tests for the pip-``sru`` drop-in (look2hear.layers.sru), its XLA kernel and the
DFT-based STFT used on XLA.

Run from the repository root:
    python -m unittest tests.test_sru -v

Tests against the pip ``sru`` package need it (and CUDA for the kernel comparison); XLA tests
need torch_xla (``PJRT_DEVICE=CPU`` is enough). Missing pieces are skipped.
"""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from look2hear.layers import dft_stft
from look2hear.layers.sru import SRU, elementwise_recurrence, run_srus_together

try:
    import torch_xla

    # Before any backward pass: the autograd engine sizes its per-device queues only once.
    XLA_DEVICE = torch_xla.device()
except ImportError:
    torch_xla = None


def _pip_sru():
    try:
        import sru

        return sru
    except Exception:
        return None


# The three SRU shapes of SwiftNet (configs/lrs2_SwiftNet_6.yml) plus the less common options.
CASES = [
    ("freq GRNN half", dict(input_size=256, hidden_size=16, num_layers=4, bidirectional=True), (58, 20)),
    ("time GRNN half", dict(input_size=32, hidden_size=32, num_layers=4), (126, 10)),
    ("video", dict(input_size=64, hidden_size=64, num_layers=4), (50, 3)),
    ("rescale+tanh", dict(input_size=24, hidden_size=12, num_layers=2, bidirectional=True,
                          rescale=True, highway_bias=-1.0, use_tanh=True), (17, 5)),
    ("layernorm+proj", dict(input_size=48, hidden_size=16, num_layers=3, layer_norm=True,
                            projection_size=8), (21, 4)),
    ("no skip", dict(input_size=20, hidden_size=10, num_layers=2, bidirectional=True,
                     has_skip_term=False), (9, 4)),
]


def _rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def _run(model, x, w, device):
    """Output, last states, input gradient and parameter gradients of sum(h * w)."""
    model = model.to(device)
    model.zero_grad()
    xi = x.detach().to(device).requires_grad_(True)
    h, c = model(xi)
    (h * w.to(device)).sum().backward()
    grads = {n: p.grad.detach().cpu() for n, p in model.named_parameters()}
    return h.detach().cpu(), c.detach().cpu(), xi.grad.detach().cpu(), grads


class TestAgainstPipSRU(unittest.TestCase):
    def setUp(self):
        self.sru = _pip_sru()
        if self.sru is None:
            self.skipTest("pip package 'sru' not importable")

    def test_same_state_dict_and_initialization(self):
        for name, kw, _ in CASES:
            with self.subTest(name):
                torch.manual_seed(0)
                ref = self.sru.SRU(**kw)
                torch.manual_seed(0)
                ours = SRU(**kw)
                sd_ref, sd_ours = ref.state_dict(), ours.state_dict()
                self.assertEqual(list(sd_ref), list(sd_ours))
                for key in sd_ref:
                    self.assertTrue(torch.equal(sd_ref[key], sd_ours[key]), key)
                # Checkpoints load in both directions.
                ours.load_state_dict(sd_ref, strict=True)
                ref.load_state_dict(sd_ours, strict=True)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_is_the_pip_kernel_and_torch_path_matches_it(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        for name, kw, (L, B) in CASES:
            with self.subTest(name):
                torch.manual_seed(0)
                ref = self.sru.SRU(**kw)
                torch.manual_seed(0)
                ours = SRU(**kw)
                x = torch.randn(L, B, kw["input_size"], generator=torch.Generator().manual_seed(1))
                w = torch.randn(L, B, ours.output_size, generator=torch.Generator().manual_seed(2))
                r = _run(ref, x, w, "cuda")
                on_cuda = _run(ours, x, w, "cuda")
                for a, b in zip(r[:3], on_cuda[:3]):
                    self.assertTrue(torch.equal(a, b))  # same kernel: bitwise
                on_cpu = _run(ours.cpu(), x, w, "cpu")  # PyTorch recurrence
                for a, b in zip(r[:3], on_cpu[:3]):
                    self.assertLess(_rel(b, a), 1e-5)
                for k in r[3]:
                    self.assertLess(_rel(on_cpu[3][k], r[3][k]), 1e-5, k)


class TestRunTogether(unittest.TestCase):
    def test_bitwise_equal_to_separate_calls(self):
        for kw, (L, B) in [(CASES[0][1], (58, 12)), (CASES[1][1], (126, 8))]:
            torch.manual_seed(3)
            a, b = SRU(**kw), SRU(**kw)
            xa = torch.randn(L, B, kw["input_size"], requires_grad=True)
            xb = torch.randn(L, B, kw["input_size"], requires_grad=True)
            params = [xa, xb, *a.parameters(), *b.parameters()]
            sep = [a(xa), b(xb)]
            g_sep = torch.autograd.grad(sum(o[0].square().sum() for o in sep), params)
            tog = run_srus_together([a, b], [xa, xb])
            g_tog = torch.autograd.grad(sum(o[0].square().sum() for o in tog), params)
            for s, t in zip(sep, tog):
                self.assertTrue(torch.equal(s[0], t[0]) and torch.equal(s[1], t[1]))
            for p, q in zip(g_sep, g_tog):
                self.assertTrue(torch.equal(p, q))


class TestDftStft(unittest.TestCase):
    def test_matches_torch_stft_and_istft(self):
        for n_fft, hop, L in [(256, 128, 32000), (256, 64, 16000), (512, 128, 24001)]:
            with self.subTest(n_fft=n_fft, hop=hop):
                w = torch.hann_window(n_fft)
                x = torch.randn(3, L, requires_grad=True)
                ref = torch.stft(x, n_fft, hop, window=w, return_complex=True)
                re, im = dft_stft.stft(x, n_fft, hop, w)
                self.assertEqual(re.shape, ref.real.shape)
                self.assertLess(max(_rel(re, ref.real), _rel(im, ref.imag)), 1e-5)
                gw = torch.randn(2, *ref.shape)
                g_ref, = torch.autograd.grad((ref.real * gw[0] + ref.imag * gw[1]).sum(), x)
                g_dft, = torch.autograd.grad((re * gw[0] + im * gw[1]).sum(), x)
                self.assertLess(_rel(g_dft, g_ref), 1e-5)

                z = torch.complex(torch.randn(ref.shape), torch.randn(ref.shape)).requires_grad_(True)
                zr = z.detach().real.clone().requires_grad_(True)
                zi = z.detach().imag.clone().requires_grad_(True)
                y_ref = torch.istft(z, n_fft, hop, window=w, length=L)
                y_dft = dft_stft.istft(zr, zi, n_fft, hop, w, L)
                self.assertEqual(y_dft.shape, y_ref.shape)
                self.assertLess(_rel(y_dft, y_ref), 1e-5)
                gy = torch.randn(3, L)
                g_z, = torch.autograd.grad((y_ref * gy).sum(), z)
                g_r, g_i = torch.autograd.grad((y_dft * gy).sum(), [zr, zi])
                self.assertLess(max(_rel(g_r, g_z.real), _rel(g_i, g_z.imag)), 1e-5)


@unittest.skipIf(torch_xla is None, "needs torch_xla")
class TestXLAKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = XLA_DEVICE

    def test_modules_match_torch_loop(self):
        for name, kw, (L, B) in CASES:
            with self.subTest(name):
                torch.manual_seed(0)
                model = SRU(**kw)
                x = torch.randn(L, B, kw["input_size"])
                w = torch.randn(L, B, model.output_size)
                ref = _run(model, x, w, "cpu")
                got = _run(model, x, w, self.device)
                for a, b in zip(got[:3], ref[:3]):
                    self.assertLess(_rel(a, b), 1e-5)
                for k in ref[3]:
                    self.assertLess(_rel(got[3][k], ref[3][k]), 1e-5, k)

    def test_recurrence_with_dropout_mask(self):
        L, B, D, d = 13, 4, 12, 6
        g = torch.Generator().manual_seed(5)
        tensors = [torch.randn(L, B, D, generator=g) for _ in range(4)]
        tensors += [torch.randn(D, generator=g) for _ in range(4)]
        tensors += [torch.randn(B, D, generator=g), (torch.rand(B, D, generator=g) > 0.3).float() / 0.7]
        for tanh in (0, 1):
            outs = []
            for device in ("cpu", self.device):
                inp = [t.detach().to(device).requires_grad_(i < 9) for i, t in enumerate(tensors)]
                h, c = elementwise_recurrence(*inp[:4], *inp[4:8], inp[8], inp[9], tanh, reverse_from=d)
                grads = torch.autograd.grad((h * torch.linspace(-1, 1, h.numel(), device=device).view_as(h)).sum()
                                            + c.square().sum(), inp[:9])
                outs.append([h, c, *grads])
            for a, b in zip(outs[1], outs[0]):
                self.assertLess(_rel(a, b), 1e-5)


if __name__ == "__main__":
    unittest.main()

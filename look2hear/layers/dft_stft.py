"""STFT / iSTFT as DFT matrix products, for devices without an FFT lowering (XLA/TPU).

PyTorch/XLA runs ``torch.stft`` / ``torch.istft`` (``aten::_fft_r2c`` / ``_fft_c2r``) on the
host CPU: every training step then splits the graph and copies data between TPU and host, in
the forward and in the backward pass. These functions compute the same transforms with
matrix products, which XLA runs on the TPU matrix units, and with plain pad/add ops for the
overlap-add.

They reproduce ``torch.stft(..., center=True, pad_mode="reflect", normalized=False,
onesided=True)`` and ``torch.istft(..., center=True, normalized=False, onesided=True,
length=...)`` up to float32 rounding (a DFT sum instead of an FFT). Matrix products run in
float32 with autocast disabled, as the FFT does on GPU.
"""

import functools
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@functools.lru_cache(maxsize=None)
def _dft_bases(n_fft: int, device: str) -> Tuple[Tensor, Tensor]:
    """Forward basis (n_fft, 2 * n_freq) = [cos | -sin] and inverse basis (2 * n_freq, n_fft).

    Built in float64 and rounded once to float32. The inverse basis is ``irfft`` with
    ``n=n_fft``: DC and Nyquist bins weigh 1/n_fft, the others 2/n_fft; the imaginary parts of
    DC and Nyquist get a zero weight (sin is 0 there), so they are ignored as by ``irfft``.
    """
    n_freq = n_fft // 2 + 1
    n = torch.arange(n_fft, dtype=torch.float64)
    k = torch.arange(n_freq, dtype=torch.float64)
    angle = 2 * math.pi * torch.outer(n, k) / n_fft  # (n_fft, n_freq)
    # Exact zeros/ones at multiples of pi/2 instead of float64 sin/cos residues.
    cos, sin = torch.cos(angle), torch.sin(angle)
    exact = (torch.outer(n, k) * 4) % n_fft == 0
    cos = torch.where(exact, cos.round(), cos)
    sin = torch.where(exact, sin.round(), sin)
    forward = torch.cat([cos, -sin], dim=1)

    weight = torch.full((n_freq,), 2.0 / n_fft, dtype=torch.float64)
    weight[0] = 1.0 / n_fft
    if n_fft % 2 == 0:
        weight[-1] = 1.0 / n_fft
    inverse = torch.cat([cos.T * weight[:, None], -sin.T * weight[:, None]], dim=0)
    return forward.float().to(device), inverse.float().to(device)


def _frames(x: Tensor, n_fft: int, hop_length: int) -> Tensor:
    """``x.unfold(-1, n_fft, hop_length)`` without ``unfold`` (an XLA host fallback).

    (batch, samples) -> (batch, frames, n_fft); needs n_fft to be a multiple of hop_length.
    """
    if n_fft % hop_length != 0:
        return x.unfold(-1, n_fft, hop_length)
    r = n_fft // hop_length
    n_frames = (x.size(-1) - n_fft) // hop_length + 1
    chunks = x[:, : (n_frames + r - 1) * hop_length].reshape(x.size(0), n_frames + r - 1, hop_length)
    # Frame t is chunks t .. t + r - 1 side by side.
    return torch.cat([chunks[:, j : j + n_frames] for j in range(r)], dim=-1)


def stft(x: Tensor, n_fft: int, hop_length: int, window: Tensor) -> Tuple[Tensor, Tensor]:
    """x: (batch, samples) -> real, imag: (batch, n_fft // 2 + 1, frames)."""
    forward, _ = _dft_bases(n_fft, str(x.device))
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        pad = n_fft // 2
        x = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        frames = _frames(x, n_fft, hop_length) * window.float()  # (batch, frames, n_fft)
        spec = torch.matmul(frames, forward)  # (batch, frames, 2 * n_freq)
    real, imag = spec.chunk(2, dim=-1)
    return real.transpose(1, 2), imag.transpose(1, 2)


def _overlap_add(frames: Tensor, hop_length: int) -> Tensor:
    """(batch, frames, n_fft) -> (batch, (frames - 1) * hop + n_fft), with n_fft a multiple of hop."""
    batch, n_frames, n_fft = frames.shape
    r = n_fft // hop_length
    chunks = frames.reshape(batch, n_frames, r, hop_length)
    out = None
    for j in range(r):
        # Frame t covers output chunks t .. t + r - 1; chunk j of frame t lands on chunk t + j.
        part = F.pad(chunks[:, :, j, :], (0, 0, j, r - 1 - j))
        out = part if out is None else out + part
    return out.reshape(batch, (n_frames + r - 1) * hop_length)


def istft(real: Tensor, imag: Tensor, n_fft: int, hop_length: int, window: Tensor, length: int) -> Tensor:
    """real, imag: (batch, n_fft // 2 + 1, frames) -> (batch, length)."""
    if n_fft % hop_length != 0:
        raise ValueError("dft_stft.istft needs n_fft to be a multiple of hop_length")
    _, inverse = _dft_bases(n_fft, str(real.device))
    with torch.autocast(device_type=real.device.type, enabled=False):
        spec = torch.cat([real.float(), imag.float()], dim=1).transpose(1, 2)  # (batch, frames, 2 * n_freq)
        frames = torch.matmul(spec, inverse) * window.float()  # (batch, frames, n_fft)
        y = _overlap_add(frames, hop_length)
        envelope = _overlap_add(window.float().square().expand(1, frames.size(1), n_fft), hop_length)
    start = n_fft // 2
    end = start + length
    y = y[:, start:end]
    envelope = envelope[:, start:end]
    y = y / envelope
    if y.size(-1) < length:
        y = F.pad(y, (0, length - y.size(-1)))
    return y

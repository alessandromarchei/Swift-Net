"""XLA-native versions of ops that PyTorch/XLA runs on the host CPU.

torch_xla 2.8 has no lowering for ``upsample_nearest`` (backward), ``_adaptive_avg_pool2d``
with non-divisible sizes, or ``im2col``/``col2im`` (``nn.Unfold``). Such ops fall back to
the CPU: the graph is cut, data goes TPU -> host -> TPU, in the forward and in the backward.

Each function here calls the original PyTorch op on any other device (so CUDA/CPU results are
unchanged) and, on XLA, computes the same values with slicing, ``index_select`` and matmuls:

* interpolate_nearest: pure copies with the source indices of PyTorch's "nearest" mode -> identical.
* unfold_kx1: pure copies in ``nn.Unfold`` order -> identical.
* adaptive_avg_pool: the same windows as PyTorch, summed then divided by the window size;
  equal up to float32 summation order.
"""

import functools
from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def _is_xla(x: Tensor) -> bool:
    return x.device.type == "xla"


@functools.lru_cache(maxsize=None)
def _nearest_index(in_size: int, out_size: int, device: str) -> Tensor:
    """Source index of each output position, as ATen's ``nearest_idx`` (legacy "nearest", size given)."""
    if out_size == in_size:
        idx = torch.arange(out_size)
    elif out_size == 2 * in_size:
        idx = torch.arange(out_size) >> 1
    else:
        scale = torch.tensor(in_size, dtype=torch.float32) / torch.tensor(out_size, dtype=torch.float32)
        idx = torch.floor(torch.arange(out_size, dtype=torch.float32) * scale).long().clamp(max=in_size - 1)
    return idx.to(device)


def interpolate_nearest(x: Tensor, size: Union[int, Sequence[int]]) -> Tensor:
    """``F.interpolate(x, size=size, mode="nearest")`` for (N, C, L) or (N, C, H, W) inputs."""
    if not _is_xla(x):
        return F.interpolate(x, size=size, mode="nearest")
    spatial = x.dim() - 2
    sizes = [int(size)] * spatial if isinstance(size, int) else [int(s) for s in size]
    for d, out_size in enumerate(sizes):
        dim = 2 + d
        if x.size(dim) != out_size:
            x = x.index_select(dim, _nearest_index(x.size(dim), out_size, str(x.device)))
    return x


@functools.lru_cache(maxsize=None)
def _pool_matrix(in_size: int, out_size: int, device: str):
    """0/1 window matrix (out, in) and window sizes (out,), with PyTorch's adaptive pooling windows."""
    rows = torch.zeros(out_size, in_size)
    counts = torch.zeros(out_size)
    for i in range(out_size):
        start = (i * in_size) // out_size
        end = ((i + 1) * in_size + out_size - 1) // out_size
        rows[i, start:end] = 1.0
        counts[i] = end - start
    return rows.to(device), counts.to(device)


def adaptive_avg_pool(x: Tensor, output_size: Sequence[int]) -> Tensor:
    """``F.adaptive_avg_pool1d/2d(x, output_size)`` for (N, C, L) or (N, C, H, W) inputs."""
    output_size = [int(s) for s in output_size]
    if not _is_xla(x):
        pool = F.adaptive_avg_pool2d if x.dim() == 4 else F.adaptive_avg_pool1d
        return pool(x, output_size)
    dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        y = x.float()
        divisor = None
        for d, out_size in enumerate(output_size):
            dim = 2 + d
            rows, counts = _pool_matrix(y.size(dim), out_size, str(x.device))
            y = torch.matmul(y.movedim(dim, -1), rows.t()).movedim(-1, dim)  # window sums
            shape = [1] * y.dim()
            shape[dim] = out_size
            counts = counts.view(shape)
            divisor = counts if divisor is None else divisor * counts
        y = y / divisor
    return y.to(dtype)


def unfold_kx1(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    """``nn.Unfold((kernel_size, 1), stride=(stride, 1))(x)`` for x (N, C, H, W)."""
    if not _is_xla(x):
        return F.unfold(x, (kernel_size, 1), stride=(stride, 1))
    N, C, H, W = x.shape
    out_h = (H - kernel_size) // stride + 1
    # Patch value at (c, k, oh, w) = x[c, oh * stride + k, w]; nn.Unfold orders channels as c * k + k_i.
    patches = torch.stack(
        [x[:, :, k : k + stride * (out_h - 1) + 1 : stride, :] for k in range(kernel_size)], dim=2
    )  # (N, C, kernel_size, out_h, W)
    return patches.reshape(N, C * kernel_size, out_h * W)

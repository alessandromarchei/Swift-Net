"""SRU (Simple Recurrent Unit) that runs on CUDA, CPU and XLA (TPU).

Drop-in replacement for ``sru.SRU`` from the pip package ``sru==2.6.0``: same constructor
arguments, same parameter names (``rnn_lst.<i>.weight`` / ``weight_c`` / ``bias`` /
``scale_x``) and the same initialization in the same RNG order. State dicts and
checkpoints are interchangeable with models built on ``sru.SRU``, and a fixed seed gives
the same initial weights.

The elementwise recurrence depends on the device of the input:

* CUDA: the fused kernel of the ``sru`` package, imported only here. GPU training is
  exactly what it was with ``sru.SRU``. Without the package, the PyTorch loop below is used.
* XLA: ``sru_xla``, forward and backward as XLA While loops built once per shape. A call is
  one node of the lazy graph instead of the whole loop traced again op by op at every step.
* CPU and anything else: a plain PyTorch loop over time (also the XLA fallback).

The XLA and CPU paths evaluate the formula of the CUDA kernel, in the same order, in float32
(as the ``sru`` CUDA path does, also under autocast):

    g1 = sigmoid(u1 + wc1 * c + b1)
    g2 = sigmoid(u2 + wc2 * c + b2)
    c  = (c - u0) * g1 + u0
    h  = (act(c) - x) * mask * g2 + x

The SRU 2.x forget gate depends on ``c`` through ``weight_c``, so the recurrence is not
linear and cannot be turned into a parallel scan: it is sequential over time on every device.
"""

import functools
import math
import warnings
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ----------------------------------------------------------------------------------------
# Elementwise recurrence
# ----------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _step_fn(activation_type: int, has_skip: bool, has_mask: bool):
    """One time step of the PyTorch loop (autograd computes its backward).

    carry: (c, wc1, wc2, b1, b2[, mask]); x: (u0, u1, u2[, skip]) at time t, each (batch, channels).
    """

    def step(carry, x):
        c, wc1, wc2, b1, b2 = carry[:5]
        u0, u1, u2 = x[0], x[1], x[2]
        g1 = torch.sigmoid(u1 + wc1 * c + b1)
        g2 = torch.sigmoid(u2 + wc2 * c + b2)
        c = (c - u0) * g1 + u0
        val = torch.tanh(c) if activation_type == 1 else c
        if has_skip:
            xv = x[3]
            diff = val - xv
            if has_mask:
                diff = diff * carry[5]
            h = diff * g2 + xv
        else:
            h = val * carry[5] * g2 if has_mask else val * g2
        return (c,) + tuple(carry[1:]), h

    return step


_XLA_KERNEL_FAILED = False


def _xla_recurrence(*args):
    """The cached XLA While-loop kernel (``sru_xla``), or None if it cannot be used here."""
    global _XLA_KERNEL_FAILED
    if _XLA_KERNEL_FAILED:
        return None
    try:
        from . import sru_xla
        return sru_xla.recurrence(*args)
    except Exception as error:  # e.g. an xla_builder API change in another torch_xla version
        _XLA_KERNEL_FAILED = True
        warnings.warn(
            "The SRU XLA kernel could not be built; falling back to an unrolled PyTorch loop, "
            f"which traces and compiles much more slowly. Cause: {error!r}"
        )
        return None


@functools.lru_cache(maxsize=None)
def _cuda_kernel():
    """The fused CUDA recurrence of the ``sru`` package, or None if it cannot be loaded."""
    try:
        from sru.ops import elementwise_recurrence_gpu
    except Exception as error:  # ImportError, or a failed JIT build of the extension
        warnings.warn(
            "The 'sru' package (fused CUDA kernel) could not be loaded; SRU falls back to a "
            f"PyTorch loop on CUDA, which is slower. Cause: {error}"
        )
        return None
    return elementwise_recurrence_gpu


def _reverse_tail(t: Tensor, start: int) -> Tensor:
    """Reverse time (dim 0) on channels [start:], the channels of the backward direction."""
    if start >= t.size(-1):
        return t
    return torch.cat([t[..., :start], t[..., start:].flip(0)], dim=-1)


def elementwise_recurrence(
    u0: Tensor,
    u1: Tensor,
    u2: Tensor,
    skip: Optional[Tensor],
    wc1: Tensor,
    wc2: Tensor,
    b1: Tensor,
    b2: Tensor,
    c0: Tensor,
    mask_c: Optional[Tensor],
    activation_type: int,
    reverse_from: int,
) -> Tuple[Tensor, Tensor]:
    """Run the SRU recurrence over time for every channel at once.

    u0, u1, u2, skip: (length, batch, channels). wc1, wc2, b1, b2: (channels,).
    c0, mask_c: (batch, channels). Channels [reverse_from:] run backward in time.
    Returns h (length, batch, channels) and the final state (batch, channels), which for the
    backward channels is the state after time step 0, as in the CUDA kernel.
    """
    xs = [_reverse_tail(t, reverse_from) for t in (u0, u1, u2)]
    if skip is not None:
        xs.append(_reverse_tail(skip, reverse_from))

    if u0.device.type == "xla":
        result = _xla_recurrence(
            xs[0], xs[1], xs[2], xs[3] if skip is not None else None,
            wc1, wc2, b1, b2, c0, mask_c, activation_type == 1,
        )
        if result is not None:
            h, c_last = result
            return _reverse_tail(h, reverse_from), c_last

    carry = (c0, wc1, wc2, b1, b2) + ((mask_c,) if mask_c is not None else ())
    step = _step_fn(activation_type, skip is not None, mask_c is not None)
    hs = []
    for t in range(u0.size(0)):
        carry, h_t = step(carry, tuple(x[t] for x in xs))
        hs.append(h_t)
    h = torch.stack(hs, dim=0)
    return _reverse_tail(h, reverse_from), carry[0]


# ----------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------

class SRUCell(nn.Module):
    """A single SRU layer, equivalent to ``sru.SRUCell`` (sru 2.6.0)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        rnn_dropout: float = 0.0,
        bidirectional: bool = False,
        n_proj: int = 0,
        use_tanh: bool = False,
        highway_bias: float = 0.0,
        has_skip_term: bool = True,
        layer_norm: bool = False,
        rescale: bool = True,
        v1: bool = False,
        custom_m: Optional[nn.Module] = None,
        amp_recurrence_fp16: bool = False,
        normalize_after: bool = False,
        weight_c_init: Optional[float] = None,
    ):
        super().__init__()
        if rnn_dropout > 0 or v1 or custom_m is not None:
            raise NotImplementedError(
                "rnn_dropout, v1 and custom_m (deprecated in sru) are not supported."
            )
        self.input_size = input_size
        self.hidden_size = hidden_size  # hidden size per direction
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_dropout = float(rnn_dropout)
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.has_skip_term = has_skip_term
        self.highway_bias = highway_bias
        self.v1 = v1
        self.rescale = rescale
        self.activation_type = 1 if use_tanh else 0
        self.activation = "tanh" if use_tanh else "none"
        self.custom_m = None
        self.amp_recurrence_fp16 = amp_recurrence_fp16
        self.normalize_after = normalize_after
        self.weight_c_init = weight_c_init

        self.projection_size = 0
        if n_proj > 0 and n_proj < self.input_size and n_proj < self.output_size:
            self.projection_size = n_proj

        self.num_matrices = 3
        if has_skip_term and self.input_size != self.output_size:
            self.num_matrices = 4

        if self.projection_size == 0:
            self.weight_proj = None
            self.weight = nn.Parameter(torch.Tensor(input_size, self.output_size * self.num_matrices))
        else:
            self.weight_proj = nn.Parameter(torch.Tensor(input_size, self.projection_size))
            self.weight = nn.Parameter(
                torch.Tensor(self.projection_size, self.output_size * self.num_matrices)
            )
        self.weight_c = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.bias = nn.Parameter(torch.Tensor(2 * self.output_size))

        # scaling constant used in highway connections when rescale=True
        self.register_buffer("scale_x", torch.FloatTensor([0]))

        self.layer_norm: Optional[nn.Module] = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.output_size if normalize_after else self.input_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Same recipe and RNG call order as ``sru.SRUCell.reset_parameters``."""
        self.bias.data.zero_()
        bias_val, output_size = self.highway_bias, self.output_size
        self.bias.data[output_size:].zero_().add_(bias_val)
        self.scale_x.data[0] = 1
        if self.rescale and self.has_skip_term:
            self.scale_x.data[0] = (1 + math.exp(bias_val) * 2) ** 0.5

        d = self.weight.size(0)
        val_range = (3.0 / d) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)
        if self.projection_size > 0:
            val_range = (3.0 / self.weight_proj.size(0)) ** 0.5
            self.weight_proj.data.uniform_(-val_range, val_range)

        # (input_size, bidirection, hidden_size, num_matrices)
        w = self.weight.data.view(d, -1, self.hidden_size, self.num_matrices)
        if self.dropout > 0:
            w[:, :, :, 0].mul_((1 - self.dropout) ** 0.5)
        if self.layer_norm:
            w.mul_(0.1)
        if self.rescale and self.has_skip_term and self.num_matrices == 4:
            w[:, :, :, 3].mul_((1 + math.exp(bias_val) * 2) ** 0.5)

        if self.weight_c_init is None:
            self.weight_c.data.uniform_(-3.0 ** 0.5, 3.0 ** 0.5)
            self.weight_c.data.mul_(0.5 ** 0.5)
        else:
            self.weight_c.data.uniform_(-self.weight_c_init, self.weight_c_init)
        w[:, :, :, 1].mul_(0.5 ** 0.5)
        w[:, :, :, 2].mul_(0.5 ** 0.5)

    def compute_U(self, input: Tensor) -> Tensor:
        """(length, batch, input_size) -> (length * batch, output_size * num_matrices)."""
        x = input.contiguous().view(-1, self.input_size)
        if self.weight_proj is not None:
            x = x.mm(self.weight_proj)
        return x.mm(self.weight)

    def get_dropout_mask_(self, size: Tuple[int, int], p: float) -> Tensor:
        return self.bias.data.new_empty(size).bernoulli_(1 - p).div_(1 - p)

    def _prepare(self, input: Tensor, c0: Tensor):
        """Everything the recurrence needs, in float32: U split per gate, skip term, dropout mask."""
        residual = input
        if self.layer_norm is not None and not self.normalize_after:
            input = self.layer_norm(input)
        length, batch = input.size(0), input.size(1)

        mask_c: Optional[Tensor] = None
        if self.training and self.dropout > 0:
            mask_c = self.get_dropout_mask_((batch, self.output_size), self.dropout)

        U = self.compute_U(input)
        u = U.view(length, batch, self.output_size, self.num_matrices).float()

        skip: Optional[Tensor] = None
        if self.has_skip_term:
            if self.num_matrices == 4:
                skip = u[..., 3]
            else:
                skip = residual.float()
                if self.rescale:
                    skip = skip * self.scale_x.float()
        return u, skip, mask_c, residual

    def forward(self, input: Tensor, c0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if input.dim() != 3:
            raise ValueError("Input must be 3 dimensional (length, batch, input_size)")
        batch = input.size(1)
        if c0 is None:
            c0 = torch.zeros(batch, self.output_size, dtype=input.dtype, device=input.device)

        if input.is_cuda:
            kernel = _cuda_kernel()
            if kernel is not None:
                return self._forward_cuda_kernel(kernel, input, c0)

        u, skip, mask_c, _ = self._prepare(input, c0)
        D = self.output_size
        h, c = elementwise_recurrence(
            u[..., 0], u[..., 1], u[..., 2], skip,
            self.weight_c[:D].float(), self.weight_c[D:].float(),
            self.bias[:D].float(), self.bias[D:].float(),
            c0.float(), None if mask_c is None else mask_c.float(),
            self.activation_type,
            reverse_from=self.hidden_size if self.bidirectional else D,
        )
        if self.layer_norm is not None and self.normalize_after:
            h = self.layer_norm(h)
        return h, c

    def _forward_cuda_kernel(self, kernel, input: Tensor, c0: Tensor) -> Tuple[Tensor, Tensor]:
        """Identical to ``sru.SRUCell.forward`` on CUDA."""
        residual = input
        if self.layer_norm is not None and not self.normalize_after:
            input = self.layer_norm(input)
        mask_c: Optional[Tensor] = None
        if self.training and self.dropout > 0:
            mask_c = self.get_dropout_mask_((input.size(1), self.output_size), self.dropout)
        scale_val = self.scale_x if self.rescale else None
        U = self.compute_U(input)
        h, c = kernel(
            U, residual, self.weight_c, self.bias, c0,
            self.activation_type, self.hidden_size, self.bidirectional,
            self.has_skip_term, scale_val, mask_c, None, self.amp_recurrence_fp16,
        )
        if self.layer_norm is not None and self.normalize_after:
            h = self.layer_norm(h)
        return h, c

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if self.projection_size > 0:
            s += ", projection_size={projection_size}"
        if self.dropout > 0:
            s += ", dropout={dropout}"
        if self.bidirectional:
            s += ", bidirectional={bidirectional}"
        if self.highway_bias != 0:
            s += ", highway_bias={highway_bias}"
        if self.activation_type != 0:
            s += ", activation={activation}"
        s += ", rescale={rescale}"
        if not self.has_skip_term:
            s += ", has_skip_term={has_skip_term}"
        if self.layer_norm:
            s += ", layer_norm=True"
        return s.format(**self.__dict__)


class SRU(nn.Module):
    """Stacked SRU, equivalent to ``sru.SRU`` (sru 2.6.0) without packed or padded input."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        rnn_dropout: float = 0.0,
        bidirectional: bool = False,
        projection_size: int = 0,
        use_tanh: bool = False,
        layer_norm: bool = False,
        highway_bias: float = 0.0,
        has_skip_term: bool = True,
        rescale: bool = False,
        v1: bool = False,
        nn_rnn_compatible_return: bool = False,
        custom_m=None,
        proj_input_to_hidden_first: bool = False,
        amp_recurrence_fp16: bool = False,
        normalize_after: bool = False,
        weight_c_init: Optional[float] = None,
    ):
        super().__init__()
        if proj_input_to_hidden_first:
            raise NotImplementedError("proj_input_to_hidden_first is not supported.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.projection_size = projection_size
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.has_skip_term = has_skip_term
        self.num_directions = 2 if bidirectional else 1
        self.nn_rnn_compatible_return = nn_rnn_compatible_return
        self.input_to_hidden = None
        self.amp_recurrence_fp16 = amp_recurrence_fp16

        self.rnn_lst = nn.ModuleList(
            SRUCell(
                input_size if i == 0 else self.output_size,
                self.hidden_size,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                n_proj=projection_size,
                use_tanh=use_tanh,
                layer_norm=layer_norm,
                highway_bias=highway_bias,
                has_skip_term=has_skip_term,
                rescale=rescale,
                v1=v1,
                custom_m=custom_m,
                amp_recurrence_fp16=amp_recurrence_fp16,
                normalize_after=normalize_after,
                weight_c_init=weight_c_init,
            )
            for i in range(num_layers)
        )

    def _initial_states(self, input: Tensor, c0: Optional[Tensor]) -> List[Tensor]:
        if c0 is None:
            zeros = torch.zeros(input.size(1), self.output_size, dtype=input.dtype, device=input.device)
            return [zeros for _ in range(self.num_layers)]
        if c0.dim() != 3:
            raise ValueError("c0 must be 3 dim (num_layers, batch_size, output_size)")
        return [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]

    def _stack_states(self, lstc: List[Tensor], batch_size: int) -> Tensor:
        lstc_stack = torch.stack(lstc)
        if self.nn_rnn_compatible_return:
            lstc_stack = lstc_stack.view(self.num_layers, batch_size, self.num_directions, self.hidden_size)
            lstc_stack = lstc_stack.transpose(1, 2).contiguous()
            lstc_stack = lstc_stack.view(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        return lstc_stack

    def forward(
        self, input: Tensor, c0: Optional[Tensor] = None, mask_pad: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """input: (length, batch, input_size) -> h (length, batch, output_size), c (num_layers, batch, output_size)."""
        if mask_pad is not None or not isinstance(input, Tensor):
            raise NotImplementedError("Padded or packed sequences are not supported.")
        if input.dim() != 3:
            raise ValueError("There must be 3 dimensions for (length, batch_size, input_size)")
        c0_ = self._initial_states(input, c0)
        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            prevx, c = rnn(prevx, c0_[i])
            lstc.append(c)
        return prevx, self._stack_states(lstc, input.size(1))


def run_srus_together(srus: Sequence[SRU], inputs: Sequence[Tensor]) -> List[Tuple[Tensor, Tensor]]:
    """Equivalent to ``[sru(x) for sru, x in zip(srus, inputs)]``, with one recurrence per layer.

    The recurrence is elementwise per channel, so SRUs with the same layout can share one time
    loop by concatenating their channels: the result is the same, with half the sequential
    steps for a pair. Used on XLA and CPU; on CUDA each SRU runs its own fused kernel.
    """
    # Other modules (e.g. the stand-ins of scripts/export_onnx.py) are simply called one by one.
    if len(srus) < 2 or inputs[0].is_cuda or not all(isinstance(s, SRU) for s in srus):
        return [sru(x) for sru, x in zip(srus, inputs)]
    first = srus[0]
    for sru, x in zip(srus, inputs):
        if (sru.num_layers, sru.bidirectional, sru.hidden_size) != (first.num_layers, first.bidirectional, first.hidden_size):
            raise ValueError("SRUs run together must have the same num_layers, bidirectional and hidden_size")
        if x.shape[:2] != inputs[0].shape[:2]:
            raise ValueError("SRUs run together need inputs with the same (length, batch)")

    d = first.hidden_size
    prevs = list(inputs)
    states = [[] for _ in srus]
    for layer in range(first.num_layers):
        cells = [sru.rnn_lst[layer] for sru in srus]
        parts = []
        for cell, x in zip(cells, prevs):
            c0 = torch.zeros(x.size(1), cell.output_size, dtype=x.dtype, device=x.device)
            parts.append((cell, c0) + cell._prepare(x, c0))

        def channels(get):
            """Concatenate one quantity of every cell: all forward channels, then all backward ones."""
            tensors = [get(p) for p in parts]
            if not first.bidirectional:
                return torch.cat(tensors, dim=-1)
            return torch.cat([t[..., :d] for t in tensors] + [t[..., d:] for t in tensors], dim=-1)

        has_skip = parts[0][3] is not None
        has_mask = parts[0][4] is not None
        if any((p[3] is not None) != has_skip or (p[4] is not None) != has_mask for p in parts):
            raise ValueError("SRUs run together must agree on skip term and dropout")
        if any(cell.activation_type != cells[0].activation_type for cell in cells):
            raise ValueError("SRUs run together must use the same activation")
        D = first.output_size
        h, c = elementwise_recurrence(
            channels(lambda p: p[2][..., 0]),
            channels(lambda p: p[2][..., 1]),
            channels(lambda p: p[2][..., 2]),
            channels(lambda p: p[3]) if has_skip else None,
            channels(lambda p: p[0].weight_c[:D].float()),
            channels(lambda p: p[0].weight_c[D:].float()),
            channels(lambda p: p[0].bias[:D].float()),
            channels(lambda p: p[0].bias[D:].float()),
            channels(lambda p: p[1].float()),
            channels(lambda p: p[4].float()) if has_mask else None,
            cells[0].activation_type,
            reverse_from=len(srus) * d if first.bidirectional else len(srus) * D,
        )

        n = len(srus)
        for i, cell in enumerate(cells):
            if first.bidirectional:
                h_i = torch.cat([h[..., i * d:(i + 1) * d], h[..., (n + i) * d:(n + i + 1) * d]], dim=-1)
                c_i = torch.cat([c[..., i * d:(i + 1) * d], c[..., (n + i) * d:(n + i + 1) * d]], dim=-1)
            else:
                h_i = h[..., i * D:(i + 1) * D]
                c_i = c[..., i * D:(i + 1) * D]
            if cell.layer_norm is not None and cell.normalize_after:
                h_i = cell.layer_norm(h_i)
            prevs[i] = h_i
            states[i].append(c_i)

    return [(prevs[i], srus[i]._stack_states(states[i], inputs[i].size(1))) for i in range(len(srus))]

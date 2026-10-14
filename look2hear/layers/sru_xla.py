"""SRU recurrence as cached XLA While loops: the TPU counterpart of the ``sru`` CUDA kernel.

Forward and backward are each one XLA computation, built once per (length, batch, channels)
and reused. A call adds a single node to the lazy graph, so the per-step tracing cost does not
grow with the sequence length, and the loop runs on the device like the CUDA kernel does.

Both loops follow the CUDA kernel of ``sru`` 2.6.0 (``sru_cuda_kernel.cu``), per channel:

    forward,  t = 0 .. L-1:
        g1 = 1 / (1 + exp(-(u1 + wc1 * c + b1)))
        g2 = 1 / (1 + exp(-(u2 + wc2 * c + b2)))
        c  = (c - u0) * g1 + u0
        h  = (act(c) - x) * mask * g2 + x            (h = act(c) * mask * g2 without skip term)

    backward, t = L-1 .. 0, with gc the gradient of c_t:
        gc  += gh * mask * g2 * act'(c_t)
        ga2  = gh * (act(c_t) - x) * mask * g2 * (1 - g2)
        ga1  = gc * (c_{t-1} - u0) * g1 * (1 - g1)
        gu0  = gc * (1 - g1);  gu1 = ga1;  gu2 = ga2;  gx = gh * (1 - mask * g2)
        gwc1 += ga1 * c_{t-1};  gwc2 += ga2 * c_{t-1};  gb1 += ga1;  gb2 += ga2
        gc   = gc * g1 + ga1 * wc1 + ga2 * wc2

Inputs are float32 tensors on an XLA device: u0, u1, u2, skip (length, batch, channels);
wc1, wc2, b1, b2 (channels,); c0, mask (batch, channels).
"""

import functools
from typing import Optional, Tuple

import torch
from torch import Tensor

import torch_xla
import torch_xla.core.xla_builder as xb

F32 = xb.Type.F32
S32 = xb.Type.S32


def _sigmoid(a):
    one = a.scalar_like(1.0)
    return one / (one + (-a).exp())


def _broadcast(vec, B, D):
    return vec.broadcast_in_dim([B, D], [1])


@functools.lru_cache(maxsize=None)
def _forward_fn(L: int, B: int, D: int, has_skip: bool, has_mask: bool, tanh: bool):
    seq, vec, mat = xb.mkshape(F32, [L, B, D]), xb.mkshape(F32, [D]), xb.mkshape(F32, [B, D])
    shapes = [seq, seq, seq] + [seq] * has_skip + [vec] * 4 + [mat] + [mat] * has_mask
    n_const = 3 + has_skip + 4 + has_mask  # loop-invariant inputs carried in the While state

    def body(t, c, h, c_all, *const):
        u0, u1, u2 = const[0], const[1], const[2]
        rest = list(const[3:])
        x = rest.pop(0) if has_skip else None
        wc1, wc2, b1, b2 = rest[0], rest[1], rest[2], rest[3]
        mask = rest[4] if has_mask else None

        zero = xb.Op.scalar(t.builder(), 0, dtype=S32)
        idx = [t, zero, zero]

        def at(seq_op):
            return seq_op.dynamic_slice(idx, [1, B, D]).reshape([B, D])

        u0t = at(u0)
        g1 = _sigmoid(at(u1) + wc1 * c + b1)
        g2 = _sigmoid(at(u2) + wc2 * c + b2)
        c = (c - u0t) * g1 + u0t
        val = c.tanh() if tanh else c
        if has_skip:
            xt = at(x)
            diff = val - xt
            if has_mask:
                diff = diff * mask
            ht = diff * g2 + xt
        else:
            ht = (val * mask if has_mask else val) * g2
        h = h.dynamic_update_slice(ht.reshape([1, B, D]), idx)
        c_all = c_all.dynamic_update_slice(c.reshape([1, B, D]), idx)
        one = xb.Op.scalar(t.builder(), 1, dtype=S32)
        return (t + one, c, h, c_all) + tuple(const)

    def cond(t, *_):
        return t < xb.Op.scalar(t.builder(), L, dtype=S32)

    def fn(*params):
        params = list(params)
        u0, u1, u2 = params[0], params[1], params[2]
        skip = params[3] if has_skip else None
        off = 3 + has_skip
        wc1, wc2, b1, b2 = (_broadcast(p, B, D) for p in params[off:off + 4])
        c0 = params[off + 4]
        mask = params[off + 5] if has_mask else None
        const = [u0, u1, u2] + [skip] * has_skip + [wc1, wc2, b1, b2] + [mask] * has_mask
        assert len(const) == n_const
        t0 = xb.Op.scalar(u0.builder(), 0, dtype=S32)
        state = [t0, c0, u0.zeros_like(), u0.zeros_like()] + const
        out = xb.Op.mkwhile(state, cond, body)
        return xb.Op.tuple([out.get_tuple_element(2), out.get_tuple_element(3)])

    key = f"L{L}_B{B}_D{D}_s{int(has_skip)}_m{int(has_mask)}_t{int(tanh)}"
    computation = xb.create_computation(f"sru_forward_{key}", fn, shapes)
    return xb.xla_computation_as_func(computation, f"xla::sru_forward_{key}")


@functools.lru_cache(maxsize=None)
def _backward_fn(L: int, B: int, D: int, has_skip: bool, has_mask: bool, tanh: bool):
    seq, vec, mat = xb.mkshape(F32, [L, B, D]), xb.mkshape(F32, [D]), xb.mkshape(F32, [B, D])
    # gh, gc_all, u0, u1, u2, [skip], wc1, wc2, b1, b2, c0, c_all, [mask]
    shapes = [seq, seq, seq, seq, seq] + [seq] * has_skip + [vec] * 4 + [mat, seq] + [mat] * has_mask
    n_grad_seq = 3 + has_skip  # gu0, gu1, gu2, [gskip]

    def body(i, gc, gwc1, gwc2, gb1, gb2, *rest):
        rest = list(rest)
        grad_seqs, rest = rest[:n_grad_seq], rest[n_grad_seq:]
        gh, gc_all, u0, u1, u2 = rest[0], rest[1], rest[2], rest[3], rest[4]
        rest2 = rest[5:]
        x = rest2.pop(0) if has_skip else None
        c_prev_all, c_all, wc1, wc2, b1, b2 = rest2[0], rest2[1], rest2[2], rest2[3], rest2[4], rest2[5]
        mask = rest2[6] if has_mask else None

        builder = i.builder()
        zero = xb.Op.scalar(builder, 0, dtype=S32)
        t = xb.Op.scalar(builder, L - 1, dtype=S32) - i
        idx = [t, zero, zero]

        def at(seq_op):
            return seq_op.dynamic_slice(idx, [1, B, D]).reshape([B, D])

        cp, ct, u0t = at(c_prev_all), at(c_all), at(u0)
        ght = at(gh)
        one = cp.scalar_like(1.0)
        g1 = _sigmoid(at(u1) + wc1 * cp + b1)
        g2 = _sigmoid(at(u2) + wc2 * cp + b2)
        val = ct.tanh() if tanh else ct

        gate = g2 * mask if has_mask else g2  # d h / d act(c)
        gv = ght * gate
        gct = gc + at(gc_all) + (gv * (one - val * val) if tanh else gv)
        if has_skip:
            xt = at(x)
            diff = val - xt
            gg2 = ght * (diff * mask if has_mask else diff)
            gxt = ght * (one - gate)
        else:
            gg2 = ght * (val * mask if has_mask else val)
        ga2 = gg2 * g2 * (one - g2)
        gu0t = gct * (one - g1)
        ga1 = gct * (cp - u0t) * g1 * (one - g1)
        gc = gct * g1 + ga1 * wc1 + ga2 * wc2

        gwc1, gwc2 = gwc1 + ga1 * cp, gwc2 + ga2 * cp
        gb1, gb2 = gb1 + ga1, gb2 + ga2
        updates = [gu0t, ga1, ga2] + ([gxt] if has_skip else [])
        grad_seqs = [g.dynamic_update_slice(v.reshape([1, B, D]), idx) for g, v in zip(grad_seqs, updates)]
        step = xb.Op.scalar(builder, 1, dtype=S32)
        return (i + step, gc, gwc1, gwc2, gb1, gb2) + tuple(grad_seqs) + tuple(rest)

    def cond(i, *_):
        return i < xb.Op.scalar(i.builder(), L, dtype=S32)

    def fn(*params):
        params = list(params)
        gh, gc_all, u0, u1, u2 = params[:5]
        skip = params[5] if has_skip else None
        off = 5 + has_skip
        wc1, wc2, b1, b2 = (_broadcast(p, B, D) for p in params[off:off + 4])
        c0, c_all = params[off + 4], params[off + 5]
        mask = params[off + 6] if has_mask else None
        # c_{t-1} for every t: c0 followed by c_0 .. c_{L-2}.
        c_prev_all = xb.Op.concat_in_dim(
            [c0.reshape([1, B, D]), c_all.slice([0, 0, 0], [L - 1, B, D])], 0
        )
        zeros_mat = c0.zeros_like()
        i0 = xb.Op.scalar(u0.builder(), 0, dtype=S32)
        state = (
            [i0, zeros_mat, zeros_mat, zeros_mat, zeros_mat, zeros_mat]
            + [u0.zeros_like() for _ in range(n_grad_seq)]
            + [gh, gc_all, u0, u1, u2] + [skip] * has_skip
            + [c_prev_all, c_all, wc1, wc2, b1, b2] + [mask] * has_mask
        )
        out = xb.Op.mkwhile(state, cond, body)

        def batch_sum(op):
            add = lambda a, b: a + b
            return op.reduce(xb.Op.scalar(op.builder(), 0.0, dtype=F32), add, [0])

        outputs = [out.get_tuple_element(6 + k) for k in range(n_grad_seq)]  # gu0, gu1, gu2, [gskip]
        outputs += [batch_sum(out.get_tuple_element(k)) for k in (2, 3, 4, 5)]  # gwc1, gwc2, gb1, gb2
        outputs.append(out.get_tuple_element(1))  # gc0
        return xb.Op.tuple(outputs)

    key = f"L{L}_B{B}_D{D}_s{int(has_skip)}_m{int(has_mask)}_t{int(tanh)}"
    computation = xb.create_computation(f"sru_backward_{key}", fn, shapes)
    return xb.xla_computation_as_func(computation, f"xla::sru_backward_{key}")


class _SRURecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tanh, u0, u1, u2, skip, wc1, wc2, b1, b2, c0, mask):
        L, B, D = u0.shape
        has_skip, has_mask = skip is not None, mask is not None
        inputs = [u0, u1, u2] + ([skip] if has_skip else []) + [wc1, wc2, b1, b2, c0] + ([mask] if has_mask else [])
        h, c_all = _forward_fn(L, B, D, has_skip, has_mask, tanh)(inputs)
        ctx.flags = (L, B, D, has_skip, has_mask, tanh)
        ctx.save_for_backward(*inputs, c_all)
        return h, c_all

    @staticmethod
    def backward(ctx, gh, gc_all):
        L, B, D, has_skip, has_mask, tanh = ctx.flags
        saved = list(ctx.saved_tensors)
        c_all = saved.pop()
        u0, u1, u2 = saved[0], saved[1], saved[2]
        skip = saved[3] if has_skip else None
        off = 3 + has_skip
        wc1, wc2, b1, b2, c0 = saved[off:off + 5]
        mask = saved[off + 5] if has_mask else None
        inputs = (
            [gh.contiguous(), gc_all.contiguous(), u0, u1, u2] + ([skip] if has_skip else [])
            + [wc1, wc2, b1, b2, c0, c_all] + ([mask] if has_mask else [])
        )
        outs = list(_backward_fn(L, B, D, has_skip, has_mask, tanh)(inputs))
        gu0, gu1, gu2 = outs[0], outs[1], outs[2]
        gskip = outs[3] if has_skip else None
        k = 3 + has_skip
        gwc1, gwc2, gb1, gb2, gc0 = outs[k:k + 5]
        return None, gu0, gu1, gu2, gskip, gwc1, gwc2, gb1, gb2, gc0, None


def recurrence(
    u0: Tensor, u1: Tensor, u2: Tensor, skip: Optional[Tensor],
    wc1: Tensor, wc2: Tensor, b1: Tensor, b2: Tensor,
    c0: Tensor, mask: Optional[Tensor], tanh: bool,
) -> Tuple[Tensor, Tensor]:
    """Returns h (length, batch, channels) and the final state (batch, channels)."""
    h, c_all = _SRURecurrence.apply(tanh, u0, u1, u2, skip, wc1, wc2, b1, b2, c0, mask)
    return h, c_all[-1]

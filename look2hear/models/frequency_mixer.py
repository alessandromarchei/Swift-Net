"""Optional convolutional frequency mixer for the FTGS block.

It replaces the frequency Bi-GSRU (``DualPathRNN`` with ``dim: 4``) and nothing else.

Causality and axes:
    * Input/output: ``[B, C, T, F]``. Every convolution has kernel ``(1, k)`` and acts on F only.
      The T axis is never mixed, so each time frame is processed independently.
    * Normalization statistics are computed per time frame (never across T), so the mixer is
      compatible with causal streaming.
    * Symmetric padding along F keeps the frequency length unchanged. Frequency is not a causal
      axis, so this is allowed.
"""

import math

import torch
import torch.nn as nn

FREQ_MIXER_TYPES = ("grnn", "conv")
BRANCH_FUSIONS = ("sum", "mean", "concat")
NORMALIZATIONS = ("layernorm", "groupnorm", "none")
ACTIVATIONS = ("prelu", "relu", "gelu", "silu")

_CONV_KEYS = {
    "kernel_sizes",
    "dilations",
    "branch_fusion",
    "hidden_channels",
    "depthwise",
    "pointwise",
    "activation",
    "normalization",
    "residual",
    "dropout",
    "num_blocks",
}


def parse_frequency_mixer_config(freq_cfg):
    """Validate the ``audio_params.frequency_mixer`` dict and return its ``type`` and ``conv`` options.

    ``None`` or a missing ``type`` means the original GRNN. Unknown keys raise an error so that
    typos in YAML do not silently fall back to defaults.
    """
    if freq_cfg is None:
        return "grnn", None

    unknown = set(freq_cfg) - {"type", "conv"}
    if unknown:
        raise ValueError(f"Unknown frequency_mixer keys: {sorted(unknown)}")

    mixer_type = freq_cfg.get("type", "grnn")
    if mixer_type not in FREQ_MIXER_TYPES:
        raise ValueError(f"frequency_mixer.type must be one of {FREQ_MIXER_TYPES}, got {mixer_type!r}")

    if mixer_type == "grnn":
        if freq_cfg.get("conv") is not None:
            raise ValueError("frequency_mixer.conv is only valid when type == 'conv'")
        return "grnn", None

    conv_cfg = dict(freq_cfg.get("conv") or {})
    unknown = set(conv_cfg) - _CONV_KEYS
    if unknown:
        raise ValueError(f"Unknown frequency_mixer.conv keys: {sorted(unknown)}")
    return "conv", conv_cfg


def _make_activation(name, channels):
    if name == "prelu":
        return nn.PReLU(channels)
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"activation must be one of {ACTIVATIONS}, got {name!r}")


class FrameLayerNorm(nn.Module):
    """LayerNorm over channels, independently at every (time, frequency) position. Input ``[B, C, T, F]``."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class FrameGroupNorm(nn.Module):
    """GroupNorm whose statistics are taken over (channels in group, F) for each (batch, time) frame."""

    def __init__(self, channels, num_groups=8, eps=1e-5):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(f"groupnorm needs channels ({channels}) divisible by {num_groups} groups")
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        B, C, T, F_ = x.shape
        g = x.reshape(B, self.num_groups, C // self.num_groups, T, F_)
        mean = g.mean(dim=(2, 4), keepdim=True)
        var = g.var(dim=(2, 4), unbiased=False, keepdim=True)
        g = ((g - mean) / torch.sqrt(var + self.eps)).reshape(B, C, T, F_)
        return g * self.weight + self.bias


def _make_norm(name, channels):
    if name == "layernorm":
        return FrameLayerNorm(channels)
    if name == "groupnorm":
        return FrameGroupNorm(channels, num_groups=math.gcd(channels, 8))
    if name == "none":
        return nn.Identity()
    raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {name!r}")


class DilatedFrequencyBlock(nn.Module):
    """One mixer block: norm -> pointwise expand -> act -> dilated freq branches -> fusion -> act -> pointwise project -> residual."""

    def __init__(
        self,
        channels,
        kernel_sizes,
        dilations,
        branch_fusion="sum",
        hidden_channels=None,
        depthwise=True,
        pointwise=True,
        activation="prelu",
        normalization="layernorm",
        residual=True,
        dropout=0.0,
    ):
        super().__init__()
        hidden = channels if hidden_channels is None else hidden_channels
        if not pointwise and hidden != channels:
            raise ValueError("pointwise: false requires hidden_channels == channels (no channel projection)")

        self.residual = residual
        self.branch_fusion = branch_fusion
        self.norm = _make_norm(normalization, channels)

        if pointwise:
            self.expand = nn.Conv2d(channels, hidden, kernel_size=1)
            self.project = nn.Conv2d(hidden, channels, kernel_size=1)
        else:
            self.expand = nn.Identity()
            self.project = nn.Identity()
        self.act_in = _make_activation(activation, hidden)

        # Frequency-only dilated convolutions, one per branch. Symmetric padding keeps F unchanged.
        self.branches = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            self.branches.append(
                nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size=(1, k),
                    dilation=(1, d),
                    padding=(0, d * (k - 1) // 2),
                    groups=hidden if depthwise else 1,
                )
            )

        # Pointwise fusion of the concatenated branches (needed to return to `hidden` channels).
        if branch_fusion == "concat":
            self.fuse = nn.Conv2d(len(self.branches) * hidden, hidden, kernel_size=1)
        else:
            self.fuse = nn.Identity()
        self.act_mid = _make_activation(activation, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        y = self.act_in(self.expand(self.norm(x)))
        outs = [branch(y) for branch in self.branches]
        if self.branch_fusion == "concat":
            y = self.fuse(torch.cat(outs, dim=1))
        else:
            y = outs[0]
            for o in outs[1:]:
                y = y + o
            if self.branch_fusion == "mean":
                y = y / len(outs)
        y = self.project(self.act_mid(y))
        y = self.dropout(y)
        return y + res if self.residual else y


class DilatedFrequencyMixer(nn.Module):
    """Frequency-only convolutional mixer. Input and output: ``[B, C, T, F]``."""

    def __init__(
        self,
        channels,
        kernel_sizes=None,
        dilations=(1, 2, 4),
        branch_fusion="sum",
        hidden_channels=None,
        depthwise=True,
        pointwise=True,
        activation="prelu",
        normalization="layernorm",
        residual=True,
        dropout=0.0,
        num_blocks=1,
    ):
        super().__init__()
        dilations = list(dilations)
        if len(dilations) == 0:
            raise ValueError("dilations must contain at least one value")
        kernel_sizes = [3] * len(dilations) if kernel_sizes is None else list(kernel_sizes)
        if len(kernel_sizes) != len(dilations):
            raise ValueError(f"len(kernel_sizes)={len(kernel_sizes)} must equal len(dilations)={len(dilations)}")
        for k in kernel_sizes:
            if k < 1 or k % 2 == 0:
                raise ValueError(f"kernel sizes must be odd and >= 1 (got {k}); even kernels are not supported")
        for d in dilations:
            if d < 1:
                raise ValueError(f"dilations must be >= 1 (got {d})")
        if branch_fusion not in BRANCH_FUSIONS:
            raise ValueError(f"branch_fusion must be one of {BRANCH_FUSIONS}, got {branch_fusion!r}")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.branch_fusion = branch_fusion
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList(
            DilatedFrequencyBlock(
                channels=channels,
                kernel_sizes=kernel_sizes,
                dilations=dilations,
                branch_fusion=branch_fusion,
                hidden_channels=hidden_channels,
                depthwise=depthwise,
                pointwise=pointwise,
                activation=activation,
                normalization=normalization,
                residual=residual,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"DilatedFrequencyMixer expects [B, C, T, F], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {x.shape[1]}")
        for block in self.blocks:
            x = block(x)
        return x


def build_frequency_module(channels, freq_cfg, grnn_builder):
    """Return the frequency modeling module for FTGS.

    ``grnn_builder`` is a zero-argument callable that builds the original ``DualPathRNN``.
    It is used only for ``type: grnn`` so the original parameter names are preserved.
    """
    mixer_type, conv_cfg = parse_frequency_mixer_config(freq_cfg)
    if mixer_type == "grnn":
        return grnn_builder()
    return DilatedFrequencyMixer(channels=channels, **conv_cfg)


def describe_frequency_module(module):
    """Return the resolved frequency-module settings as a flat dict, for startup logs and diagnostics.

    Works for both the original GRNN (``DualPathRNN``) and ``DilatedFrequencyMixer``.
    """
    info = {"module": type(module).__name__, "params": sum(p.numel() for p in module.parameters())}
    if not isinstance(module, DilatedFrequencyMixer):
        return info
    block = module.blocks[0]
    info.update(
        {
            "branch_fusion": module.branch_fusion,
            "num_blocks": module.num_blocks,
            "num_branches": len(block.branches),
            "kernel_sizes": list(module.kernel_sizes),
            "dilations": list(module.dilations),
            "hidden_channels": block.expand.out_channels if isinstance(block.expand, nn.Conv2d) else module.channels,
            "has_fuse_conv": isinstance(block.fuse, nn.Conv2d),
        }
    )
    return info

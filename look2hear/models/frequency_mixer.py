"""Optional frequency mixers (convolutional or self-attention) for the FTGS block.

It replaces the frequency Bi-GSRU (``DualPathRNN`` with ``dim: 4``) and nothing else.

Causality and axes:
    * Input/output: ``[B, C, T, F]``. Every convolution has kernel ``(1, k)`` and acts on F only.
      The T axis is never mixed, so each time frame is processed independently.
    * Normalization statistics are computed per time frame (never across T), so the mixer is
      compatible with causal streaming.
    * Symmetric padding along F keeps the frequency length unchanged. Frequency is not a causal
      axis, so this is allowed.
    * The attention mixer attends over F (or ERB bands) inside each frame, never across T.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FREQ_MIXER_TYPES = ("grnn", "conv", "mhsa")
POS_ENCODINGS = ("none", "sinusoidal", "learned", "rope")
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

_ATTENTION_KEYS = {
    "num_layers",
    "n_head",
    "embed_dim",
    "ffn_dim",
    "pos_enc",
    "max_len",
    "erb_bands",
    "sample_rate",
    "dropout",
    "pre_conv_kernel",
    "pre_conv_depthwise",
    "unfold_kernel",
    "unfold_stride",
    "pos_base",
}


def parse_frequency_mixer_config(freq_cfg):
    """Validate the ``audio_params.frequency_mixer`` dict and return its ``type`` and the options of that type.

    ``None`` or a missing ``type`` means the original GRNN. Unknown keys raise an error so that
    typos in YAML do not silently fall back to defaults.
    """
    if freq_cfg is None:
        return "grnn", None

    unknown = set(freq_cfg) - {"type", "conv", "mhsa"}
    if unknown:
        raise ValueError(f"Unknown frequency_mixer keys: {sorted(unknown)}")

    mixer_type = freq_cfg.get("type", "grnn")
    if mixer_type not in FREQ_MIXER_TYPES:
        raise ValueError(f"frequency_mixer.type must be one of {FREQ_MIXER_TYPES}, got {mixer_type!r}")

    # Only the sub-dict matching `type` may be set
    for other in ("conv", "mhsa"):
        if other != mixer_type and freq_cfg.get(other) is not None:
            raise ValueError(f"frequency_mixer.{other} is only valid when type == '{other}'")

    if mixer_type == "grnn":
        return "grnn", None

    allowed = _CONV_KEYS if mixer_type == "conv" else _ATTENTION_KEYS
    sub_cfg = dict(freq_cfg.get(mixer_type) or {})
    unknown = set(sub_cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown frequency_mixer.{mixer_type} keys: {sorted(unknown)}")
    return mixer_type, sub_cfg


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


# ---------------------------------------------------------------------------------------------
# Self-attention frequency mixer
# ---------------------------------------------------------------------------------------------


def erb_band_matrices(n_freqs, n_bands, sample_rate=16000):
    """Rectangular ERB bands over ``n_freqs`` linear bins (0 .. sample_rate / 2).

    Returns ``compress`` ``[n_bands, n_freqs]`` (average of the bins of each band) and
    ``expand`` ``[n_freqs, n_bands]`` (copy each band value back to its bins).
    """
    if not 1 <= n_bands <= n_freqs:
        raise ValueError(f"erb_bands must be in [1, n_freqs={n_freqs}], got {n_bands}")

    def hz_to_erb(f):
        return 21.4 * math.log10(1.0 + 0.00437 * f)

    def erb_to_hz(e):
        return (10 ** (e / 21.4) - 1.0) / 0.00437

    # Band edges equally spaced on the ERB scale, converted to bin indices
    nyquist = sample_rate / 2
    top = hz_to_erb(nyquist)
    edges = [round(erb_to_hz(top * i / n_bands) / nyquist * n_freqs) for i in range(n_bands + 1)]
    edges[0], edges[-1] = 0, n_freqs
    # Every band gets at least one bin (low bands are narrower than one bin otherwise)
    for i in range(1, n_bands):
        edges[i] = max(edges[i], edges[i - 1] + 1)
    for i in range(n_bands - 1, 0, -1):
        edges[i] = min(edges[i], edges[i + 1] - 1)

    expand = torch.zeros(n_freqs, n_bands)
    for b in range(n_bands):
        expand[edges[b] : edges[b + 1], b] = 1.0
    compress = expand.t() / expand.sum(dim=0, keepdim=True).t()
    return compress, expand


def sinusoidal_encoding(length, dim, device=None, base=10000.0):
    """Classic transformer sinusoidal table ``[length, dim]``. ``base`` sets the slowest wavelength."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(base) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def apply_rope(x, base=10000.0):
    """Rotary position embedding on ``[N, heads, L, head_dim]`` (head_dim must be even)."""
    L, D = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    angles = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1) * inv_freq  # [L, D/2]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class FrequencyAttentionLayer(nn.Module):
    """Pre-norm transformer layer on ``[N, L, D]``: multi-head self-attention (+ optional FFN)."""

    def __init__(self, dim, n_head, ffn_dim=0, dropout=0.0, rope=False, rope_base=10000.0):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f"embed_dim ({dim}) must be divisible by n_head ({n_head})")
        if rope and (dim // n_head) % 2 != 0:
            raise ValueError(f"rope needs an even head dim, got embed_dim / n_head = {dim // n_head}")
        self.n_head = n_head
        self.rope = rope
        self.rope_base = rope_base
        self.dropout = dropout

        self.norm_attn = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        if ffn_dim > 0:
            self.norm_ffn = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        else:
            self.ffn = None

    def forward(self, x):
        N, L, D = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).reshape(N, L, 3, self.n_head, D // self.n_head).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = apply_rope(q, self.rope_base), apply_rope(k, self.rope_base)
        # No mask: frequency is not a causal axis, every bin sees every bin of the same frame
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(N, L, D)
        x = x + self.drop(self.out(y))
        if self.ffn is not None:
            x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return x


class FrequencyConv1d(nn.Module):
    """Pre-norm residual 1D convolution along the token axis of ``[N, L, D]`` (L = frequency bins or ERB bands)."""

    def __init__(self, dim, kernel_size, depthwise=True):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"pre_conv_kernel must be odd and >= 1 (got {kernel_size})")
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim if depthwise else 1)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.act(y)


class FrequencyUnfold(nn.Module):
    """Patch tokens along frequency, like the ``nn.Unfold`` of the frequency Bi-GSRU.

    ``tokens``: ``[N, L, C]`` -> ``[N, L', embed_dim]``. Token ``i`` concatenates the C channels of the
    ``kernel_size`` bins starting at ``i * stride`` (zero padded so that bin ``f`` is the centre of
    token ``f`` when stride is 1), then a linear layer projects ``C * kernel_size`` to ``embed_dim``.
    ``fold``: ``[N, L', embed_dim]`` -> ``[N, L, C]`` with a transposed convolution, i.e. every token
    writes back to all the bins of its patch and the overlaps are summed.
    """

    def __init__(self, channels, embed_dim, kernel_size, stride=1):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"unfold_kernel must be >= 1 (got {kernel_size})")
        if not 1 <= stride <= kernel_size:
            raise ValueError(f"unfold_stride must be in [1, unfold_kernel={kernel_size}] (got {stride})")
        self.kernel_size = kernel_size
        self.stride = stride
        self.proj = nn.Linear(channels * kernel_size, embed_dim)
        self.fold_conv = nn.ConvTranspose1d(embed_dim, channels, kernel_size, stride=stride)

    def _padding(self, L):
        left = (self.kernel_size - 1) // 2
        # Token i is centred on bin i * stride; add one more token if the last bins are not covered
        n_tokens = (L - 1) // self.stride + 1
        if (n_tokens - 1) * self.stride - left + self.kernel_size < L:
            n_tokens += 1
        # Right padding so that the last patch ends exactly on the padded length
        right = (n_tokens - 1) * self.stride + self.kernel_size - left - L
        return left, right

    def tokens(self, x):
        left, right = self._padding(x.shape[1])
        x = F.pad(x.transpose(1, 2), (left, right))  # [N, C, L + pad]
        x = x.unfold(-1, self.kernel_size, self.stride)  # [N, C, L', k]
        return self.proj(x.permute(0, 2, 1, 3).flatten(2))  # [N, L', C * k] -> [N, L', embed_dim]

    def fold(self, y, L):
        left, _ = self._padding(L)
        y = self.fold_conv(y.transpose(1, 2))  # [N, C, L + pad]
        return y[..., left : left + L].transpose(1, 2)


class AttentionFrequencyMixer(nn.Module):
    """Frequency self-attention mixer. Input and output: ``[B, C, T, F]``.

    Each time frame is processed independently: its F bins (or ERB bands) are the tokens and
    the C channels are the token features. Steps:
        [optional] ERB compression F -> erb_bands
        linear C -> embed_dim, or with unfold_kernel > 0: patches of unfold_kernel bins
            (C * unfold_kernel features, stride unfold_stride) projected to embed_dim
        [optional] residual conv1d along the tokens (pre_conv_kernel > 0)
        + positional embedding (sinusoidal / learned; rope acts inside attention)
        num_layers x (MHSA [+ FFN])
        linear embed_dim -> C, or with unfolding: overlap-add fold back to every bin
        [optional] ERB expansion erb_bands -> F
        + residual
    """

    def __init__(
        self,
        channels,
        num_layers=1,
        n_head=4,
        embed_dim=None,
        ffn_dim=0,
        pos_enc="rope",
        max_len=512,
        erb_bands=None,
        sample_rate=16000,
        dropout=0.0,
        pre_conv_kernel=0,
        pre_conv_depthwise=True,
        unfold_kernel=0,
        unfold_stride=1,
        pos_base=10000.0,
    ):
        super().__init__()
        if pos_enc not in POS_ENCODINGS:
            raise ValueError(f"pos_enc must be one of {POS_ENCODINGS}, got {pos_enc!r}")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        token_dim = channels * unfold_kernel if unfold_kernel > 0 else channels
        # Without an explicit embed_dim the tokens keep their own size (no projection cost saving)
        embed_dim = token_dim if embed_dim is None else embed_dim

        self.channels = channels
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.num_layers = num_layers
        self.pos_enc = pos_enc
        self.pos_base = float(pos_base)
        self.erb_bands = erb_bands
        self.sample_rate = sample_rate
        self._erb_cache = {}  # n_freqs -> (compress, expand), built on first use

        self.unfold_kernel = unfold_kernel
        self.unfold_stride = unfold_stride
        if unfold_kernel > 0:
            self.unfold = FrequencyUnfold(channels, embed_dim, unfold_kernel, unfold_stride)
            self.in_proj = self.out_proj = None
        else:
            self.unfold = None
            self.in_proj = nn.Linear(channels, embed_dim) if embed_dim != channels else nn.Identity()
            self.out_proj = nn.Linear(embed_dim, channels) if embed_dim != channels else nn.Identity()
        # Local mixing of neighbouring bins before attention; 0 disables it
        self.pre_conv_kernel = pre_conv_kernel
        self.pre_conv = FrequencyConv1d(embed_dim, pre_conv_kernel, pre_conv_depthwise) if pre_conv_kernel > 0 else None

        if pos_enc == "learned":
            # Sequence length is erb_bands when ERB is on, otherwise the number of bins (<= max_len)
            self.pos_emb = nn.Parameter(torch.zeros(erb_bands or max_len, embed_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.layers = nn.ModuleList(
            FrequencyAttentionLayer(
                embed_dim, n_head, ffn_dim=ffn_dim, dropout=dropout, rope=pos_enc == "rope", rope_base=self.pos_base
            )
            for _ in range(num_layers)
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def _erb(self, n_freqs, x):
        if n_freqs not in self._erb_cache:
            self._erb_cache[n_freqs] = erb_band_matrices(n_freqs, self.erb_bands, self.sample_rate)
        compress, expand = self._erb_cache[n_freqs]
        return compress.to(x), expand.to(x)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"AttentionFrequencyMixer expects [B, C, T, F], got {tuple(x.shape)}")
        B, C, T, F_ = x.shape
        residual = x

        # [B, C, T, F] -> tokens [B*T, L, C]
        y = x.permute(0, 2, 3, 1).reshape(B * T, F_, C)
        if self.erb_bands:
            compress, expand = self._erb(F_, y)
            y = torch.matmul(compress, y)  # [B*T, erb_bands, C]

        n_tokens = y.shape[1]  # F, or erb_bands
        y = self.unfold.tokens(y) if self.unfold is not None else self.in_proj(y)
        if self.pre_conv is not None:
            y = self.pre_conv(y)
        L = y.shape[1]
        if self.pos_enc == "sinusoidal":
            y = y + sinusoidal_encoding(L, self.embed_dim, device=y.device, base=self.pos_base).to(y.dtype)
        elif self.pos_enc == "learned":
            if L > self.pos_emb.shape[0]:
                raise ValueError(f"sequence length {L} exceeds learned pos_emb length {self.pos_emb.shape[0]} (raise max_len)")
            y = y + self.pos_emb[:L]

        for layer in self.layers:
            y = layer(y)
        y = self.norm_out(y)
        y = self.unfold.fold(y, n_tokens) if self.unfold is not None else self.out_proj(y)

        if self.erb_bands:
            y = torch.matmul(expand, y)  # [B*T, F, C]

        y = y.reshape(B, T, F_, C).permute(0, 3, 1, 2)
        return y + residual


def build_frequency_module(channels, freq_cfg, grnn_builder):
    """Return the frequency modeling module for FTGS.

    ``grnn_builder`` is a zero-argument callable that builds the original ``DualPathRNN``.
    It is used only for ``type: grnn`` so the original parameter names are preserved.
    """
    mixer_type, mixer_cfg = parse_frequency_mixer_config(freq_cfg)
    if mixer_type == "grnn":
        return grnn_builder()
    if mixer_type == "mhsa":
        return AttentionFrequencyMixer(channels=channels, **mixer_cfg)
    return DilatedFrequencyMixer(channels=channels, **mixer_cfg)


def describe_frequency_module(module):
    """Return the resolved frequency-module settings as a flat dict, for startup logs and diagnostics.

    Works for the original GRNN (``DualPathRNN``), ``DilatedFrequencyMixer`` and ``AttentionFrequencyMixer``.
    """
    info = {"module": type(module).__name__, "params": sum(p.numel() for p in module.parameters())}
    if isinstance(module, AttentionFrequencyMixer):
        info.update(
            {
                "num_layers": module.num_layers,
                "n_head": module.n_head,
                "embed_dim": module.embed_dim,
                "pos_enc": module.pos_enc,
                "erb_bands": module.erb_bands,
                "pre_conv_kernel": module.pre_conv_kernel,
                "unfold_kernel": module.unfold_kernel,
                "unfold_stride": module.unfold_stride,
                "pos_base": module.pos_base,
            }
        )
        return info
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

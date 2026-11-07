"""TinyVisualEncoder: the small visual frontend distilled in the TinyVisualEncoder repo, as a SwiftNet video model.

Drop-in replacement for FRCNNVideoModel (same call signature, same output layout):

    input : [B, 1, T, H, W] mouth crops, normalized by the SwiftNet pipeline (LRW mean/std, 88x88)
    output: [B, embedding_dim, T]

The architecture (``TinyVisualFrontend``) is a copy of
TinyVisualEncoder/models/tiny_visual_frontend.py with identical module names, so the KD checkpoints
(``<run>/checkpoints/best.pt``, key ``model``) load with strict=True. KD training code stays in the
TinyVisualEncoder repo; only the inference architecture lives here.

The student was distilled on 112x112 grayscale crops normalized with (x/255 - 0.4161)/0.1688. The
SwiftNet mouth pipeline produces (x/255 - 0.421)/0.165 at 88x88, so ``forward`` maps the input back to
the student's normalization (exact affine) and resizes it to ``input_size`` before the frontend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1):
        p = k // 2
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class DSBlock(nn.Module):
    """Depthwise-separable residual block."""

    def __init__(self, cin, cout, stride=1, expand=2):
        super().__init__()
        hidden = max(cin, int(cin * expand))
        self.block = nn.Sequential(
            ConvBNAct(cin, hidden, 1, 1),
            ConvBNAct(hidden, hidden, 3, stride, groups=hidden),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = (
            nn.Identity()
            if cin == cout and stride == 1
            else nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))


class TinyVisualFrontend(nn.Module):
    """[T, B, 1, H, W] -> [T, B, D]. Only the first Conv3d mixes time; the rest is frame-wise.

    ``causal=True`` pads the temporal axis with ``temporal_kernel - 1`` past frames instead of symmetric
    padding (same convention as FRCNNVideoModel). The KD weights were trained non-causally.
    """

    def __init__(
        self,
        embedding_dim=512,
        temporal_kernel=5,
        stem_channels=16,
        widths=(24, 32, 64, 96),
        expand=2,
        causal=False,
    ):
        super().__init__()
        if temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be odd")

        self.embedding_dim = embedding_dim
        self.temporal_kernel = temporal_kernel
        self.causal = causal

        self.frontend3d = nn.Sequential(
            nn.Conv3d(
                1,
                stem_channels,
                kernel_size=(temporal_kernel, 5, 5),
                stride=(1, 2, 2),
                padding=(0 if causal else temporal_kernel // 2, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(stem_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        blocks = []
        cin = stem_channels
        for cout in widths:
            blocks.append(DSBlock(cin, cout, stride=2, expand=expand))
            blocks.append(DSBlock(cout, cout, stride=1, expand=expand))
            cin = cout

        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(cin, embedding_dim)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [T,B,C,H,W], got {tuple(x.shape)}")

        x = x.permute(1, 2, 0, 3, 4).contiguous()  # [B,C,T,H,W]
        if self.causal:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_kernel - 1, 0))
        x = self.frontend3d(x)

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        x = x.reshape(B, T, self.embedding_dim)
        return x.permute(1, 0, 2).contiguous()  # [T,B,D]


def _parse_widths(widths):
    if isinstance(widths, str):
        return tuple(int(w) for w in widths.split(",") if w.strip())
    return tuple(int(w) for w in widths)


class TinyVisualEncoderModel(nn.Module):
    """SwiftNet video model wrapping ``TinyVisualFrontend``: [B, 1, T, H, W] -> [B, D, T].

    Args:
        embedding_dim, temporal_kernel, stem_channels, widths, expand: student architecture. Must match
            the KD run that produced ``pretrain`` (checked against the checkpoint ``args``).
        pretrain: KD checkpoint (``checkpoints/best.pt`` of a TinyVisualEncoder run). Loaded strictly.
        input_size: spatial size fed to the frontend (the KD resolution). null: keep the input size.
        input_mean, input_std: normalization already applied by the dataset pipeline.
        target_mean, target_std: normalization the student was trained with.
        causal: past-only temporal padding in the first Conv3d.
        freeze_bn: keep BatchNorm layers in eval mode during training (running stats not updated).
            With a frozen encoder this must be true, otherwise ``model.train()`` would silently update
            the running stats.
    """

    def __init__(
        self,
        embedding_dim=512,
        temporal_kernel=5,
        stem_channels=16,
        widths=(24, 32, 64, 96),
        expand=2.0,
        pretrain=None,
        input_size=112,
        input_mean=0.421,
        input_std=0.165,
        target_mean=0.4161,
        target_std=0.1688,
        causal=False,
        freeze_bn=True,
    ):
        super().__init__()
        self.widths = _parse_widths(widths)
        self.expand = expand
        self.input_size = None if input_size is None else int(input_size)
        # x_target = x_input * scale + shift  (exact change of normalization)
        self.norm_scale = float(input_std) / float(target_std)
        self.norm_shift = (float(input_mean) - float(target_mean)) / float(target_std)
        self.freeze_bn = bool(freeze_bn)
        self.backend_out = int(embedding_dim)

        self.frontend = TinyVisualFrontend(
            embedding_dim=int(embedding_dim),
            temporal_kernel=int(temporal_kernel),
            stem_channels=int(stem_channels),
            widths=self.widths,
            expand=expand,
            causal=causal,
        )

        self.pretrain = pretrain
        if pretrain:
            self.init_from(pretrain)

    def init_from(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        args = ckpt.get("args") if isinstance(ckpt, dict) else None
        if args:
            expected = {
                "embedding_dim": self.frontend.embedding_dim,
                "temporal_kernel": self.frontend.temporal_kernel,
                "stem_channels": self.frontend.frontend3d[0].out_channels,
                "widths": self.widths,
                "expand": float(self.expand),
            }
            saved = {
                "embedding_dim": args.get("embedding_dim"),
                "temporal_kernel": args.get("temporal_kernel"),
                "stem_channels": args.get("stem_channels"),
                "widths": _parse_widths(args["widths"]) if "widths" in args else None,
                "expand": float(args["expand"]) if "expand" in args else None,
            }
            diff = {k: (saved[k], expected[k]) for k in expected if saved[k] is not None and saved[k] != expected[k]}
            if diff:
                raise ValueError(f"TinyVisualEncoder config does not match the checkpoint {path}: {diff}")
        self.frontend.load_state_dict(state, strict=True)

    def forward(self, x):
        # x: [B, 1, T, H, W] (FRCNNVideoModel layout)
        if x.ndim == 4:
            x = x.unsqueeze(1)
        B, C, T, H, W = x.shape
        x = x * self.norm_scale + self.norm_shift
        x = x.transpose(1, 2)  # [B, T, C, H, W]
        if self.input_size is not None and (H, W) != (self.input_size, self.input_size):
            x = F.interpolate(
                x.reshape(B * T, C, H, W),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, T, C, self.input_size, self.input_size)
        x = x.transpose(0, 1)  # [T, B, C, h, w]
        y = self.frontend(x)  # [T, B, D]
        return y.permute(1, 2, 0).contiguous()  # [B, D, T]

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()
        return self

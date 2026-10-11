#!/usr/bin/env python3
"""Profile one SwiftNet training step on the GPU, without CUPTI (kernel-level profilers are unavailable here).

Three views:
  1. time per step component (forward, backward, optimizer), CUDA events, median over steps;
  2. forward and backward time per module, inclusive, summed over repeated calls within a step
     (the FTGS block runs `repeats` times, so its time is the sum of all its calls);
  3. GPU time per ATen operator for one forward+backward step: a TorchDispatchMode puts CUDA events
     around every top-level op; events are synchronized at the end of the step, so each op is timed alone.
     This serializes execution, so per-op numbers are exact but the sum is slightly above the normal step time.

Usage:
    python scripts/profile_swiftnet.py --conf_dir configs/lrs2_SwiftNet_6.yml --batch_size 3 \
        --output results/profile_swiftnet_grnn.txt
"""
import argparse
import collections
import copy
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from torch.utils._python_dispatch import TorchDispatchMode

from look2hear.models import SwiftNet


class OpTimer(TorchDispatchMode):
    """Records CUDA events around each dispatched ATen op. Nested ops are not intercepted by the mode."""

    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = func(*args, **kwargs)
        e.record()
        self.records.append((str(func), s, e))
        return out


def modules_to_profile(net):
    sep = net.separator
    named = {
        "encoder (STFT + conv)": net.encoder,
        "audio_bottleneck": net.audio_bottleneck,
        "video_bottleneck": net.video_bottleneck,
        "separator.audio_net (FTGS, all repeats)": sep.audio_net,
        "  FTGS.process_f (frequency GRNN)": sep.audio_net.process_f,
        "  FTGS.process_t (temporal GRNN)": sep.audio_net.process_t,
        "  FTGS.process_att (causal MHSA 2D)": sep.audio_net.process_att,
        "  FTGS.downsample_layers (all)": list(sep.audio_net.downsample_layers),
        "  FTGS.fusion_layers (InjectionMultiSum, all)": list(sep.audio_net.fusion_layers),
        "  FTGS.concat_layers (InjectionMultiSum, all)": list(sep.audio_net.concat_layers),
        "  FTGS.gateway": sep.audio_net.gateway,
        "  FTGS.projection": sep.audio_net.projection,
        "  FTGS.residual_conv": sep.audio_net.residual_conv,
        "separator.video_net (LightVid)": sep.video_net,
        "separator.caf (SAF fusion)": sep.caf,
        "mask_generator": net.mask_generator,
        "decoder (iSTFT)": net.decoder,
    }
    return {k: v for k, v in named.items() if v is not None}


def median(values):
    return statistics.median(values) if values else float("nan")


def main(args):
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    with open(args.conf_dir) as f:
        conf = yaml.safe_load(f)
    net = SwiftNet(sample_rate=16000, **copy.deepcopy(conf["audionet"]["audionet_config"])).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
    B = args.batch_size
    wav = torch.randn(B, 32000, device=device)
    emb = torch.randn(B, 512, 50, device=device)
    target = torch.randn(B, 1, 32000, device=device)

    def timed_pair():
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        return s, e

    def step(store=None):
        s, e = timed_pair()
        est = net(wav, emb)
        loss = (est - target).pow(2).mean()
        e.record()
        if store is not None:
            store["forward"].append((s, e))
        s, e = timed_pair()
        loss.backward()
        e.record()
        if store is not None:
            store["backward"].append((s, e))
        s, e = timed_pair()
        opt.step()
        opt.zero_grad(set_to_none=True)
        e.record()
        if store is not None:
            store["optimizer"].append((s, e))

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    # 1+2) component timings and per-leaf-module timings over the whole network.
    # Every leaf module gets forward and backward hooks, so the leaves cover the entire model.
    leaves = {name: m for name, m in net.named_modules() if name and not list(m.children())}
    current = collections.defaultdict(lambda: collections.defaultdict(list))
    handles = []

    def add_hooks(name, m):
        def fpre(mod, inp):
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            mod._t_fstart = s
        def fpost(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            current[name]["forward"].append((mod._t_fstart, e))
        def bpre(mod, grad_out):
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            mod._t_bstart = s
        def bpost(mod, grad_in, grad_out):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            current[name]["backward"].append((mod._t_bstart, e))
        handles.append(m.register_forward_pre_hook(fpre))
        handles.append(m.register_forward_hook(fpost))
        handles.append(m.register_full_backward_pre_hook(bpre))
        handles.append(m.register_full_backward_hook(bpost))

    for name, m in leaves.items():
        add_hooks(name, m)

    steps = {"forward": [], "backward": [], "optimizer": []}
    per_leaf = {name: {"forward": [], "backward": []} for name in leaves}
    for _ in range(args.iters):
        current.clear()
        step(steps)
        torch.cuda.synchronize()
        for name in leaves:
            for kind in ("forward", "backward"):
                pairs = current[name][kind]
                per_leaf[name][kind].append(sum(s.elapsed_time(e) for s, e in pairs))
    for h in handles:
        h.remove()

    def med(values):
        return statistics.median(values)

    comp = {k: med([s.elapsed_time(e) for s, e in v]) for k, v in steps.items()}
    total = sum(comp.values())
    leaf_fwd = {n: med(v["forward"]) for n, v in per_leaf.items()}
    leaf_bwd = {n: med(v["backward"]) for n, v in per_leaf.items()}
    cov_fwd, cov_bwd = sum(leaf_fwd.values()), sum(leaf_bwd.values())

    def section(name):
        # Top-level section: the first two components of the module path (e.g. separator.audio_net)
        parts = name.split(".")
        return ".".join(parts[:2]) if parts[0] == "separator" else parts[0]

    def type_of(name):
        return type(leaves[name]).__name__

    lines = [
        f"SwiftNet training step profile (whole network) | config {args.conf_dir} | batch {B} | fp32 | GPU {torch.cuda.get_device_name(0)}",
        f"Trainable params: {sum(p.numel() for p in net.parameters()):,} | leaf modules: {len(leaves)} | steps measured: {args.iters}",
        "",
        "1) Component time per training step (median ms)",
    ]
    for k, v in comp.items():
        lines.append(f"   {v:9.2f} ms  {k:10s} ({100 * v / total:4.1f}%)")
    lines.append(f"   {total:9.2f} ms  total")
    lines.append(f"   coverage: leaf modules account for {cov_fwd:.1f} ms of {comp['forward']:.1f} ms forward "
                 f"and {cov_bwd:.1f} ms of {comp['backward']:.1f} ms backward")
    lines.append("")

    def table(title, groups):
        lines.append(title)
        lines.append(f"   {'':52s} {'forward':>10s} {'backward':>10s} {'total':>10s} {'share':>7s}")
        ranked = sorted(groups.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
        grand = cov_fwd + cov_bwd
        for key, (f_ms, b_ms) in ranked[: args.top]:
            lines.append(f"   {key[:52]:52s} {f_ms:10.2f} {b_ms:10.2f} {f_ms + b_ms:10.2f} {100 * (f_ms + b_ms) / grand:6.1f}%")
        lines.append("")

    by_section = collections.defaultdict(lambda: [0.0, 0.0])
    by_type = collections.defaultdict(lambda: [0.0, 0.0])
    for n in leaves:
        by_section[section(n)][0] += leaf_fwd[n]
        by_section[section(n)][1] += leaf_bwd[n]
        by_type[type_of(n)][0] += leaf_fwd[n]
        by_type[type_of(n)][1] += leaf_bwd[n]
    table("2) By top-level section (ms per step, median)", by_section)
    table("3) By module type (ms per step, median)", by_type)
    table("4) Individual leaf modules (ms per step, median)",
          {n: (leaf_fwd[n], leaf_bwd[n]) for n in leaves})

    # 3) ATen op GPU time for one forward+backward+optimizer step
    timer = OpTimer()
    with timer:
        s0, e0 = timed_pair()
        est = net(wav, emb)
        loss = (est - target).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        e0.record()
    torch.cuda.synchronize()
    agg = collections.defaultdict(lambda: [0.0, 0])
    for name, s, e in timer.records:
        agg[name][0] += s.elapsed_time(e)
        agg[name][1] += 1
    grand = sum(v[0] for v in agg.values())
    lines.append(f"3) ATen ops by GPU time, one step ({len(timer.records)} ops, serialized sum {grand:.1f} ms)")
    lines.append(f"   {'op':52s} {'ms':>10s} {'share':>7s} {'calls':>7s}")
    for name, (ms, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[: args.top]:
        lines.append(f"   {name[:52]:52s} {ms:10.2f} {100 * ms / grand:6.1f}% {n:7d}")

    text = "\n".join(lines)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\nWritten {out}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conf_dir", default="configs/lrs2_SwiftNet_6.yml")
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))

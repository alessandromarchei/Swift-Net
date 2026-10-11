#!/usr/bin/env python3
"""Benchmark the original frequency GRNN against the convolutional frequency mixer.

Three variants are compared on the SAME input tensor (shape read from the real SwiftNet frequency stage):
    grnn        original frequency Bi-GSRU (DualPathRNN, dim 4) from configs/lrs2_SwiftNet_6.yml
    conv_sum    DilatedFrequencyMixer, branch_fusion: sum   (configs/lrs2_SwiftNet_6_freqmix_sum.yml)
    conv_concat DilatedFrequencyMixer, branch_fusion: concat (configs/lrs2_SwiftNet_6_freqmix_concat.yml)

Per variant and precision (fp32, tf32, bf16 autocast) it reports median/IQR latency of forward, backward and
forward+backward (CUDA events), throughput, peak GPU memory, trainable parameters, and MACs from
torch.utils.flop_counter. MACs are only counted for standard ops; the SRU kernels are invisible to the counter,
so the GRNN MAC number is marked as partial.

With --full_model it also times a complete SwiftNet training step (forward, loss, backward, AdamW step) for
each variant, using random audio and embeddings, so the isolated mixer speedup can be compared with the end-to-end one.

Usage:
    python scripts/benchmark_frequency_mixer.py --device cuda --iters 100 --output results/freq_bench.json
"""
import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from look2hear.models import SwiftNet
from look2hear.models.frequency_mixer import build_frequency_module
from look2hear.models.SwiftNet import DualPathRNN

VARIANTS = {
    "grnn": "configs/lrs2_SwiftNet_6.yml",
    "conv_sum": "configs/lrs2_SwiftNet_6_freqmix_sum.yml",
    "conv_concat": "configs/lrs2_SwiftNet_6_freqmix_concat.yml",
}
PRECISIONS = ("fp32", "tf32", "bf16")


def load_audionet_config(path):
    with open(_resolve(path)) as f:
        return yaml.safe_load(f)["audionet"]["audionet_config"]


def _resolve(p):
    return p if Path(p).is_absolute() else str(_REPO_ROOT / p)


def set_precision(precision):
    """fp32: TF32 off. tf32: TF32 on for matmul/conv. bf16: TF32 on and bf16 autocast in the forward."""
    use_tf32 = precision in ("tf32", "bf16")
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")


def autocast_ctx(precision, device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def summarize(values):
    values = sorted(values)
    n = len(values)
    q = statistics.quantiles(values, n=4) if n >= 4 else [values[0], values[0], values[-1]]
    return {
        "median": statistics.median(values),
        "p25": q[0],
        "p75": q[2],
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def count_macs(module, x, precision, device):
    """MACs of one forward pass. Counted only for standard ops (conv/linear/matmul)."""
    with torch.no_grad(), autocast_ctx(precision, device), FlopCounterMode(display=False) as fc:
        module(x)
    return fc.get_total_flops() // 2


def time_module(module, x, precision, device, warmup, iters):
    """Latency of forward and backward of a standalone frequency module (CUDA events on GPU)."""
    use_events = device.type == "cuda"
    module.train()

    def step(record):
        if use_events:
            s0, s1, s2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            s0.record()
        else:
            t0 = time.perf_counter()
        with autocast_ctx(precision, device):
            out = module(x)
            loss = out.float().pow(2).mean()
        if use_events:
            s1.record()
        else:
            t1 = time.perf_counter()
        loss.backward()
        if use_events:
            s2.record()
            torch.cuda.synchronize()
            fwd_ms, bwd_ms = s0.elapsed_time(s1), s1.elapsed_time(s2)
        else:
            t2 = time.perf_counter()
            fwd_ms, bwd_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3
        module.zero_grad(set_to_none=True)
        return fwd_ms, bwd_ms

    for _ in range(warmup):
        step(record=False)

    if use_events:
        torch.cuda.synchronize(device)
        base_mem = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    fwd, bwd = [], []
    for _ in range(iters):
        f, b = step(record=True)
        fwd.append(f)
        bwd.append(b)
    peak_mb = (torch.cuda.max_memory_allocated(device) - base_mem) / 2**20 if use_events else None
    return fwd, bwd, peak_mb


def capture_frequency_shape(conf_path, batch, audio_samples, device):
    """Run one forward pass through the real SwiftNet and read the shape entering the frequency stage."""
    ac = load_audionet_config(conf_path)
    net = SwiftNet(sample_rate=16000, **copy.deepcopy(ac)).to(device).eval()
    seen = {}
    handle = net.separator.audio_net.process_f.register_forward_hook(
        lambda mod, inp, out: seen.update(inp=tuple(inp[0].shape), out=tuple(out.shape))
    )
    with torch.no_grad():
        net(torch.randn(batch, audio_samples, device=device), torch.randn(batch, 512, 50, device=device))
    handle.remove()
    return seen["inp"], seen["out"]


def build_freq_module(conf_path, channels, device):
    ac = load_audionet_config(conf_path)
    layers = copy.deepcopy(ac["audio_params"]["layers"])
    freq_cfg = ac["audio_params"].get("frequency_mixer")
    module = build_frequency_module(
        channels=channels,
        freq_cfg=freq_cfg,
        grnn_builder=lambda: DualPathRNN(in_chan=channels, **layers["layer_1"]),
    )
    return module.to(device)


def full_model_step_time(conf_path, batch, audio_samples, precision, device, warmup, iters):
    """Latency of a full SwiftNet training step (forward, loss, backward, AdamW step)."""
    use_events = device.type == "cuda"
    ac = load_audionet_config(conf_path)
    net = SwiftNet(sample_rate=16000, **copy.deepcopy(ac)).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
    wav = torch.randn(batch, audio_samples, device=device)
    emb = torch.randn(batch, 512, 50, device=device)
    target = torch.randn(batch, 1, audio_samples, device=device)

    def step():
        if use_events:
            e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            e[0].record()
        else:
            t0 = time.perf_counter()
        with autocast_ctx(precision, device):
            est = net(wav, emb)
        loss = (est.float() - target).pow(2).mean()
        if use_events:
            e[1].record()
        loss.backward()
        if use_events:
            e[2].record()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if use_events:
            e[3].record()
            torch.cuda.synchronize()
            return e[0].elapsed_time(e[3])
        return (time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        step()
    if use_events:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
    times = [step() for _ in range(iters)]
    peak = (torch.cuda.max_memory_allocated(device) - base) / 2**20 if use_events else None
    params = sum(p.numel() for p in net.parameters())
    return summarize(times), peak, params


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    (in_shape, out_shape) = capture_frequency_shape(VARIANTS["grnn"], args.batch_size, args.audio_samples, device)
    B, C, T, F_ = in_shape
    print(f"Frequency stage input (from SwiftNet forward): {in_shape}, output {out_shape}")
    print(f"Device: {device} {torch.cuda.get_device_name(device) if device.type == 'cuda' else ''}")

    results = {"input_shape": list(in_shape), "device": str(device), "variants": {}}
    for name, conf in VARIANTS.items():
        results["variants"][name] = {"module": {}, "full_model": {}}
        module = build_freq_module(conf, C, device)
        params = sum(p.numel() for p in module.parameters())
        x = torch.randn(B, C, T, F_, device=device)
        macs = count_macs(module, x, "fp32", device)
        partial = name == "grnn"
        results["variants"][name]["params"] = params
        results["variants"][name]["macs_forward"] = macs
        results["variants"][name]["macs_partial"] = partial
        print(f"\n== {name}: params={params:,} MACs(fwd)={macs:,}{' (partial, SRU not counted)' if partial else ''}")

        for precision in args.precisions:
            set_precision(precision)
            try:
                fwd, bwd, peak = time_module(module, x, precision, device, args.warmup, args.iters)
            except RuntimeError as err:  # e.g. unsupported bf16 kernel
                results["variants"][name]["module"][precision] = {"error": str(err)[:200]}
                print(f"  {precision}: unsupported ({str(err)[:120]})")
                continue
            total = [f + b for f, b in zip(fwd, bwd)]
            entry = {
                "forward_ms": summarize(fwd),
                "backward_ms": summarize(bwd),
                "forward_backward_ms": summarize(total),
                "throughput_samples_per_s": B / (summarize(total)["median"] / 1e3),
                "peak_gpu_mem_mb": peak,
            }
            results["variants"][name]["module"][precision] = entry
            print(
                f"  {precision:5s} fwd {entry['forward_ms']['median']:.3f} ms  bwd {entry['backward_ms']['median']:.3f} ms  "
                f"fwd+bwd {entry['forward_backward_ms']['median']:.3f} ms "
                f"(IQR {entry['forward_backward_ms']['p25']:.3f}-{entry['forward_backward_ms']['p75']:.3f})  "
                f"throughput {entry['throughput_samples_per_s']:.0f} samples/s  peak {peak:.1f} MiB"
                if peak is not None
                else f"  {precision:5s} fwd+bwd {entry['forward_backward_ms']['median']:.3f} ms (CPU)"
            )

        if args.full_model:
            for precision in args.precisions:
                set_precision(precision)
                try:
                    stats, peak, net_params = full_model_step_time(
                        conf, args.batch_size, args.audio_samples, precision, device, args.warmup, args.iters
                    )
                except RuntimeError as err:
                    results["variants"][name]["full_model"][precision] = {"error": str(err)[:200]}
                    print(f"  full model {precision}: unsupported ({str(err)[:120]})")
                    continue
                results["variants"][name]["full_model"][precision] = {
                    "step_ms": stats,
                    "peak_gpu_mem_mb": peak,
                    "params": net_params,
                }
                print(f"  full SwiftNet step {precision:5s}: median {stats['median']:.2f} ms "
                      f"(IQR {stats['p25']:.2f}-{stats['p75']:.2f}), params {net_params:,}")
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nWritten {out}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=3, help="Same as datamodule batch_size in the configs")
    p.add_argument("--audio_samples", type=int, default=32000, help="2 s at 16 kHz")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=list(PRECISIONS))
    p.add_argument("--full_model", action="store_true", help="Also time a full SwiftNet training step")
    p.add_argument("--output", default=None, help="Write the results as JSON")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))

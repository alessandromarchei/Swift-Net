#!/usr/bin/env python3
"""End-to-end training-step benchmark for three configurations on identical data.

    A  original SwiftNet-6 (GRNN) + online visual backbone        configs/lrs2_SwiftNet_6.yml
    B  original SwiftNet-6 (GRNN) + cached visual embeddings      configs/lrs2_SwiftNet_6_cached.yml
    C  SwiftNet-6 with conv frequency mixer (sum) + cached        configs/lrs2_SwiftNet_6_freqmix_sum.yml

All variants use the same train split (shuffle off, drop_last), batch size, segment length, precision,
device, DataLoader workers and loss. Each step is split into phases, timed with torch.cuda.synchronize()
around each phase:
    data        DataLoader next() + host-to-device copy
    visual      FRCNNVideoModel forward (A only; 0 for cached variants)
    forward     audio model forward + loss
    backward    loss.backward()
    optimizer   AdamW step + zero_grad
Warm-up steps are excluded. Reported values are median, IQR and std over the measured steps.

Batch identity: the sample names of every measured batch are hashed and compared across variants, so
a mismatch in the data order would be reported.

Usage:
    python scripts/benchmark_training_pipeline.py --device cuda --steps 60 --warmup 10 --output results/train_bench.json
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

import look2hear.losses
import look2hear.models
import look2hear.videomodels
from look2hear.datas import AVSpeechDataModule

VARIANTS = {
    "A_original_online": {"conf": "configs/lrs2_SwiftNet_6.yml", "cached": False},
    "B_original_cached": {"conf": "configs/lrs2_SwiftNet_6_cached.yml", "cached": True},
    "C_freqmix_cached": {"conf": "configs/lrs2_SwiftNet_6_freqmix_sum.yml", "cached": True},
}


def resolve(p):
    return p if Path(p).is_absolute() else str(_REPO_ROOT / p)


def summarize(values):
    values = sorted(values)
    q = statistics.quantiles(values, n=4)
    return {"median": statistics.median(values), "p25": q[0], "p75": q[2],
            "std": statistics.pstdev(values), "mean": statistics.fmean(values)}


def set_precision(precision):
    use_tf32 = precision in ("tf32", "bf16")
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")


def run_variant(name, spec, args, device):
    with open(resolve(spec["conf"])) as f:
        conf = yaml.safe_load(f)
    dc = dict(conf["datamodule"]["data_config"])
    dc["visual_features_path"] = args.cache_dir if spec["cached"] else None
    if args.num_workers is not None:
        dc["num_workers"] = args.num_workers
    if args.batch_size is not None:
        dc["batch_size"] = args.batch_size  # same batch size for every variant
    if spec["cached"] and not Path(args.cache_dir).exists():
        raise FileNotFoundError(f"cache directory not found: {args.cache_dir}")

    datamodule = AVSpeechDataModule(**dc)
    datamodule.setup()
    loader = DataLoader(
        datamodule.data_train,
        batch_size=dc["batch_size"],
        shuffle=False,  # identical batches for every variant
        num_workers=dc["num_workers"],
        pin_memory=dc["pin_memory"] and device.type == "cuda",
        persistent_workers=dc["persistent_workers"] and dc["num_workers"] > 0,
        drop_last=True,
    )

    ac = conf["audionet"]["audionet_config"]
    audio = getattr(look2hear.models, conf["audionet"]["audionet_name"])(
        sample_rate=dc["sample_rate"], **ac
    ).to(device).train()
    video = None
    if not spec["cached"]:
        video = getattr(look2hear.videomodels, conf["videonet"]["videonet_name"])(
            **conf["videonet"]["videonet_config"]
        ).to(device).eval()
        for p in video.parameters():
            p.requires_grad_(False)

    loss_cfg = conf["loss"]["train"]
    loss_fn = getattr(look2hear.losses, loss_cfg["loss_func"])(
        getattr(look2hear.losses, loss_cfg["sdr_type"]), **loss_cfg["config"]
    )
    opt = torch.optim.AdamW(audio.parameters(), lr=conf["optimizer"]["lr"],
                            weight_decay=conf["optimizer"]["weight_decay"])

    use_cuda = device.type == "cuda"

    def sync():
        if use_cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    phases = {k: [] for k in ("data", "visual", "forward", "backward", "optimizer")}
    batch_hashes = []
    total_steps = args.warmup + args.steps
    it = iter(loader)
    peak_mb = None
    for step in range(total_steps):
        if step == args.warmup and use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            base_mem = torch.cuda.memory_allocated(device)

        t = sync()
        mixtures, targets, mouth, names = next(it)
        mixtures = mixtures.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mouth = mouth.to(device, non_blocking=True)
        t_data = sync() - t

        t = sync()
        if spec["cached"]:
            emb = mouth.float()  # precomputed [B, D, T]
        else:
            m = mouth.unsqueeze(1) if mouth.ndim == 4 else mouth
            with torch.no_grad():
                emb = video(m.type_as(mixtures))
        t_visual = sync() - t

        if targets.ndim == 2:
            targets = targets.unsqueeze(1)
        t = sync()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            est = audio(mixtures, emb)
        loss = loss_fn(est, targets)
        t_fwd = sync() - t

        t = sync()
        loss.backward()
        t_bwd = sync() - t

        t = sync()
        opt.step()
        opt.zero_grad(set_to_none=True)
        t_opt = sync() - t

        if step >= args.warmup:
            for key, val in zip(phases, (t_data, t_visual, t_fwd, t_bwd, t_opt)):
                phases[key].append(val * 1e3)
            batch_hashes.append(hashlib.sha256("|".join(names).encode()).hexdigest()[:12])
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite loss at step {step}")

    if use_cuda:
        peak_mb = (torch.cuda.max_memory_allocated(device) - base_mem) / 2**20

    total = [sum(vals) for vals in zip(*phases.values())]
    stats = {k: summarize(v) for k, v in phases.items()}
    stats["total"] = summarize(total)
    result = {
        "phases_ms": stats,
        "samples_per_s": dc["batch_size"] / (stats["total"]["median"] / 1e3),
        "peak_gpu_mem_mb": peak_mb,
        "batch_size": dc["batch_size"],
        "num_workers": dc["num_workers"],
        "trainable_params": sum(p.numel() for p in audio.parameters()),
        "batch_hashes": batch_hashes,
    }
    del loader, it, audio, video, opt
    if use_cuda:
        torch.cuda.empty_cache()
    return result


def main(args):
    device = torch.device(args.device)
    set_precision(args.precision)
    print(f"Device: {device} | precision: {args.precision} | warmup {args.warmup} | steps {args.steps}")
    results = {}
    for name in args.variants:
        print(f"\n== {name}")
        r = run_variant(name, VARIANTS[name], args, device)
        results[name] = r
        ph = r["phases_ms"]
        print(f"  data {ph['data']['median']:.2f} | visual {ph['visual']['median']:.2f} | forward {ph['forward']['median']:.2f} | "
              f"backward {ph['backward']['median']:.2f} | optimizer {ph['optimizer']['median']:.2f} | "
              f"total {ph['total']['median']:.2f} ms (IQR {ph['total']['p25']:.2f}-{ph['total']['p75']:.2f})")
        print(f"  throughput {r['samples_per_s']:.1f} samples/s"
              + (f" | peak GPU {r['peak_gpu_mem_mb']:.0f} MiB" if r["peak_gpu_mem_mb"] is not None else ""))

    hashes = {name: r["batch_hashes"] for name, r in results.items()}
    same = len({tuple(h) for h in hashes.values()}) == 1
    print(f"\nIdentical measured batches across variants: {same}")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"precision": args.precision, "device": str(device),
                                   "batches_identical": same, "results": results}, indent=2))
        print(f"Written {out}")
    return 0 if same else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--cache_dir", default="DataPreProcess/LRS2_visual_features")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--precision", default="fp32", choices=["fp32", "tf32", "bf16"])
    p.add_argument("--num_workers", type=int, default=None, help="Override; default is the config value for every variant")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override the config batch size. The LRS2 config (4) exceeds GPU memory for SwiftNet training on a 16 GB card; 3 fits.")
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))

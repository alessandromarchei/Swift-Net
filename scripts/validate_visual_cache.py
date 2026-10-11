#!/usr/bin/env python3
"""Numerical equivalence: online visual backbone vs. cached visual embeddings.

For the same validation samples (eval mode, same deterministic preprocessing) it reports:
  * embedding error: online FRCNNVideoModel vs. the cache (max abs, mean abs, mean relative);
  * separator output error: audio_model(mix, online_emb) vs. audio_model(mix, cached_emb);
  * FP16 quantization error of the embeddings (round trip fp32 -> fp16 -> fp32);
  * for reference, how far the training-time augmentation (RandomCrop + HorizontalFlip) moves the
    embedding. This is the part that a cache cannot reproduce.

Exit code 1 if an FP32 cache deviates beyond the tolerance, so mismatches are never ignored.

Usage:
    python scripts/validate_visual_cache.py --conf_dir configs/lrs2_SwiftNet_6.yml \
        --cache_dir DataPreProcess/LRS2_visual_features --split cv --num_samples 64 \
        --ckpt Experiments/LRS2_SwiftNet_6/checkpoints/last.ckpt
"""
import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import yaml

import look2hear.models
import look2hear.videomodels
from look2hear.datas.avspeech_dataset import AVSpeechDataset
from look2hear.datas.visual_cache import VisualFeatureCache
from look2hear.utils.checkpoint_utils import load_audio_weights


def stats(a, b):
    """Error statistics between reference ``a`` and test ``b`` (same shape)."""
    diff = (a - b).abs()
    denom = a.abs().mean().clamp_min(1e-12)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "mean_rel": (diff.mean() / denom).item(),
    }


def snr_db(reference, estimate):
    """SNR of ``estimate`` against ``reference`` in dB (higher is closer)."""
    err = (reference - estimate).pow(2).sum()
    sig = reference.pow(2).sum().clamp_min(1e-12)
    return (10 * torch.log10(sig / err.clamp_min(1e-30))).item()


def load_split(conf, split, cache, is_train=False):
    dc = conf["datamodule"]["data_config"]
    json_dir = {"tr": dc["train_dir"], "cv": dc["valid_dir"], "tt": dc["test_dir"]}[split]
    return AVSpeechDataset(
        json_dir=json_dir,
        n_src=dc["n_src"],
        sample_rate=dc["sample_rate"],
        segment=dc["segment"],
        normalize_audio=dc["normalize_audio"],
        is_train=is_train,
        visual_cache=cache,
    )


def main(args):
    torch.manual_seed(0)
    with open(args.conf_dir) as f:
        conf = yaml.safe_load(f)
    device = torch.device(args.device)

    cache = VisualFeatureCache(args.cache_dir)
    online_ds = load_split(conf, args.split, None)
    cached_ds = load_split(conf, args.split, cache)
    aug_ds = load_split(conf, args.split, None, is_train=True)

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(online_ds), size=min(args.num_samples, len(online_ds)), replace=False)

    videonet = conf["videonet"]
    video_model = getattr(look2hear.videomodels, videonet["videonet_name"])(**videonet["videonet_config"])
    video_model.eval().to(device)

    audio_model = getattr(look2hear.models, conf["audionet"]["audionet_name"])(
        sample_rate=conf["datamodule"]["data_config"]["sample_rate"],
        **conf["audionet"]["audionet_config"],
    ).eval()
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        load_audio_weights(audio_model, ckpt["state_dict"], mode="strict")
        print(f"Audio model loaded from {args.ckpt}")
    else:
        print("No --ckpt: audio model has random weights (embedding errors are still exact).")
    audio_model.to(device)

    emb_online_all, emb_cached_all, emb_aug_all, mixes = [], [], [], []
    est_online_all, est_cached_all = [], []

    for start in range(0, len(indices), args.batch_size):
        idx = indices[start : start + args.batch_size]
        online_items = [online_ds[int(i)] for i in idx]
        cached_items = [cached_ds[int(i)] for i in idx]
        # The online dataset returns NumPy mouths for n_src=1 (as before the cache existed).
        mouths = torch.stack([torch.as_tensor(np.asarray(item[2])) for item in online_items]).float()  # [B, T, 88, 88]
        with torch.inference_mode():
            emb_online = video_model(mouths.unsqueeze(1).to(device)).cpu()  # [B, D, T]
        emb_cached = torch.stack([item[2] for item in cached_items]).float()
        if emb_cached.shape != emb_online.shape:
            raise RuntimeError(f"shape mismatch: cached {tuple(emb_cached.shape)} vs online {tuple(emb_online.shape)}")

        # Training-time augmentation for reference only: same source, random crop and flip.
        aug = []
        for i in idx:
            random.seed(int(i))
            aug.append(torch.as_tensor(np.asarray(aug_ds[int(i)][2])))
        emb_aug = []
        with torch.inference_mode():
            emb_aug = video_model(torch.stack(aug).float().unsqueeze(1).to(device)).cpu()

        mix = torch.stack([torch.as_tensor(np.asarray(item[0])) for item in online_items]).float()
        with torch.inference_mode():
            est_online = audio_model(mix.to(device), emb_online.to(device)).cpu()
            est_cached = audio_model(mix.to(device), emb_cached.to(device)).cpu()

        emb_online_all.append(emb_online)
        emb_cached_all.append(emb_cached)
        emb_aug_all.append(emb_aug)
        est_online_all.append(est_online)
        est_cached_all.append(est_cached)
        mixes.append(mix)

    emb_online = torch.cat(emb_online_all)
    emb_cached = torch.cat(emb_cached_all)
    emb_aug = torch.cat(emb_aug_all)
    est_online = torch.cat(est_online_all)
    est_cached = torch.cat(est_cached_all)

    print(f"\nSamples: {len(indices)} | split: {args.split} | cache dtype: {cache.metadata['feature_dtype']}")
    print(f"Embedding shape per sample: {tuple(emb_online.shape[1:])}")

    fp32_emb = stats(emb_online, emb_cached)
    fp16_roundtrip = stats(emb_online, emb_online.half().float())
    aug = stats(emb_online, emb_aug)
    out = stats(est_online, est_cached)
    snr = snr_db(est_online, est_cached)

    print("\n[1] Embedding error: online backbone vs. cache")
    for k, v in fp32_emb.items():
        print(f"    {k:9s} {v:.3e}")
    print("[2] FP16 storage round trip of the online embeddings (quantization only)")
    for k, v in fp16_roundtrip.items():
        print(f"    {k:9s} {v:.3e}")
    print("[3] Separator output: audio_model(mix, online) vs. audio_model(mix, cached)")
    for k, v in out.items():
        print(f"    {k:9s} {v:.3e}")
    print(f"    SNR of cached output relative to online output: {snr:.1f} dB")
    print("[4] Reference only: training augmentation (random crop + flip) vs. deterministic crop")
    for k, v in aug.items():
        print(f"    {k:9s} {v:.3e}")

    ok = True
    if cache.metadata["feature_dtype"] == "float32":
        ok = fp32_emb["max_abs"] <= args.atol and out["max_abs"] <= args.output_atol
        print(
            f"\nFP32 check: embedding max_abs <= {args.atol:g} and output max_abs <= {args.output_atol:g}: "
            + ("PASS" if ok else "FAIL")
        )
    else:
        # FP16 storage: round-to-nearest error is at most 2^-11 * |x| per element (10 mantissa bits).
        bound = 2 ** -11 * emb_online.abs().max().item() + args.atol
        ok = fp32_emb["max_abs"] <= bound
        print(f"\nFP16 check: embedding max_abs <= half-precision bound {bound:.3e}: " + ("PASS" if ok else "FAIL"))
    print(
        "\nNote: the cache is deterministic (center crop, no flip). Online training uses RandomCrop and "
        "HorizontalFlip, so cached training is NOT equivalent to online training with augmentation. "
        "See [4] for the size of that difference."
    )
    return 0 if ok else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conf_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--split", default="cv", choices=["tr", "cv", "tt"])
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=None, help="Lightning checkpoint with trained audio weights (optional)")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--atol", type=float, default=1e-4, help="FP32 tolerance on embeddings (absolute)")
    p.add_argument("--output_atol", type=float, default=1e-3, help="FP32 tolerance on separator output (absolute)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))

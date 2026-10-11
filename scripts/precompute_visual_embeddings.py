#!/usr/bin/env python3
"""Precompute FRCNNVideoModel embeddings once and store them in a resumable cache.

Usage:
    python scripts/precompute_visual_embeddings.py \
        --conf_dir configs/lrs2_SwiftNet_6.yml \
        --output_dir DataPreProcess/LRS2_visual_features \
        --splits tr cv tt --batch_size 64 --num_workers 8

Preprocessing (deterministic, val pipeline of look2hear.datas.transform):
    Normalize(0, 255) -> CenterCrop(88x88) -> Normalize(0.421, 0.165) -> first fps_len frames.
    The training pipeline in the dataset additionally uses RandomCrop and HorizontalFlip(0.5).
    Those two steps are stochastic and cannot be cached: see the README/report for the impact.

Entries are keyed by the mouth .npz path of each visual source. A utterance that appears in several
mixtures is computed once. Re-running the script skips completed entries and recomputes the rest.
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import look2hear.videomodels
from look2hear.datas.avspeech_dataset import AVSpeechDataset
from look2hear.datas.transform import get_preprocessing_pipelines
from look2hear.datas.visual_cache import (
    CACHE_FORMAT_VERSION,
    VisualFeatureCache,
    check_metadata,
    create_cache,
    normalize_visual_key,
    verify_cache,
    write_rows,
)

FPS = 25  # visual frame rate of the mouth crops (dataset uses int(segment * 25) frames)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_metadata(videonet_name, videonet_config, feature_dim, num_frames, segment, feature_dtype):
    pretrain = videonet_config.get("pretrain")
    backbone = {
        "videonet_name": videonet_name,
        "config": {k: v for k, v in videonet_config.items() if k != "pretrain"},
        "pretrain_path": pretrain,
        "pretrain_sha256": sha256_file(pretrain) if pretrain else None,
    }
    preprocessing = {
        "name": "val_center_crop",
        "stochastic": False,
        "steps": ["Normalize(0, 255)", "CenterCrop(88, 88)", "Normalize(0.421, 0.165)", "frames[:fps_len]"],
    }
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "backbone": backbone,
        "preprocessing": preprocessing,
        "input_fps": FPS,
        "segment_duration": segment,
        "num_frames": num_frames,
        "feature_dim": feature_dim,
        "feature_dtype": feature_dtype,
    }


class PendingMouths(Dataset):
    """Loads and preprocesses mouth files that still need an embedding."""

    def __init__(self, rows, paths, num_frames):
        self.rows = rows
        self.paths = paths
        self.num_frames = num_frames
        self.pipeline = get_preprocessing_pipelines()["val"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        frames = self.pipeline(np.load(self.paths[i])["data"])[:, : self.num_frames]
        return self.rows[i], torch.from_numpy(frames).float()


def collate_keep_lengths(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def collect_keys(conf, splits):
    """Every visual source used by the datasets of the requested splits, using the dataset's own drop rules."""
    dc = conf["datamodule"]["data_config"]
    split_dirs = {"tr": dc["train_dir"], "cv": dc["valid_dir"], "tt": dc["test_dir"]}
    keys = set()
    for split in splits:
        ds = AVSpeechDataset(
            json_dir=split_dirs[split],
            n_src=dc["n_src"],
            sample_rate=dc["sample_rate"],
            segment=dc["segment"],
            normalize_audio=dc["normalize_audio"],
            is_train=False,
        )
        keys.update(ds.visual_keys())
    return sorted(keys)


def compute_embeddings(video_model, mouths, device):
    """Run the backbone on a list of [T, 88, 88] tensors. Returns a list of [D, T] float32 CPU tensors."""
    lengths = {m.shape[0] for m in mouths}
    with torch.inference_mode():
        if len(lengths) == 1:
            x = torch.stack(mouths).unsqueeze(1).to(device)  # B, 1, T, H, W
            emb = video_model(x).float().cpu()  # B, D, T
            return list(emb.unbind(0))
        # Variable lengths are rare (short sources): process one by one, same math.
        out = []
        for m in mouths:
            out.append(video_model(m[None, None].to(device)).float().cpu()[0])
        return out


def main(args):
    with open(args.conf_dir) as f:
        conf = yaml.safe_load(f)

    dc = conf["datamodule"]["data_config"]
    videonet = conf["videonet"]
    num_frames = int(dc["segment"] * FPS)
    feature_dim = 512
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    expected = build_metadata(
        videonet["videonet_name"],
        videonet["videonet_config"],
        feature_dim,
        num_frames,
        dc["segment"],
        args.dtype,
    )

    keys = collect_keys(conf, args.splits)
    print(f"Visual sources in splits {args.splits}: {len(keys)}")

    root = args.output_dir
    if os.path.exists(os.path.join(root, "metadata.json")):
        existing = VisualFeatureCache(root)
        check_metadata(expected, existing.metadata)
        missing_from_index = sorted(set(keys) - set(existing.keys))
        if missing_from_index:
            raise RuntimeError(
                f"{len(missing_from_index)} visual sources are not in the existing cache index {root}. "
                "Use a new --output_dir for a different set of splits."
            )
        print(f"Resuming cache {root}: {len(existing)} entries indexed")
    else:
        create_cache(root, keys, dict(expected, splits=args.splits))
        print(f"Created cache {root}")

    if args.verify_only:
        summary = verify_cache(root)
        print_summary(summary)
        return 0 if not summary["non_finite_rows"] and not summary["invalid_length_rows"] else 1

    cache = VisualFeatureCache(root)
    rows_needed = [cache.row[normalize_visual_key(k)] for k in keys]
    pending = [r for r in rows_needed if cache.status[r] != 1]
    print(f"Pending: {len(pending)} / {len(rows_needed)} (already completed: {len(rows_needed) - len(pending)})")

    if pending:
        paths = [cache.keys[r] for r in pending]
        video_model = getattr(look2hear.videomodels, videonet["videonet_name"])(**videonet["videonet_config"])
        video_model.eval()
        for p in video_model.parameters():
            p.requires_grad_(False)
        video_model.to(device)

        loader = DataLoader(
            PendingMouths(pending, paths, num_frames),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_keep_lengths,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

        np_dtype = np.dtype(args.dtype)
        done, start = 0, time.time()
        for batch_idx, (batch_rows, mouths) in enumerate(loader):
            emb = compute_embeddings(video_model, mouths, device)
            n = len(batch_rows)
            block = np.zeros((n, feature_dim, num_frames), dtype=np_dtype)
            lens = np.zeros(n, dtype=np.uint16)
            for j, e in enumerate(emb):
                t = e.shape[-1]
                if t > num_frames:
                    raise RuntimeError(
                        f"embedding has {t} frames but the cache slot holds {num_frames}; "
                        "check segment and the cache metadata"
                    )
                block[j, :, :t] = e.numpy().astype(np_dtype)
                lens[j] = t
            # Flush features before status: an interrupted run recomputes unfinished rows only.
            write_rows(root, np.asarray(batch_rows), block, lens)
            done += n
            if batch_idx % 20 == 0 or done == len(pending):
                rate = done / max(time.time() - start, 1e-6)
                print(f"  {done}/{len(pending)} ({rate:.1f} sources/s)")

    summary = verify_cache(root)
    print_summary(summary)
    return 0 if not summary["non_finite_rows"] and not summary["invalid_length_rows"] and not summary["pending"] else 1


def print_summary(summary):
    print(
        f"Cache: {summary['completed']} completed, {summary['pending']} pending, "
        f"{len(summary['non_finite_rows'])} non-finite, {len(summary['invalid_length_rows'])} invalid length "
        f"(of {summary['num_entries']})"
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conf_dir", required=True, help="YAML config (uses videonet and datamodule sections)")
    p.add_argument("--output_dir", required=True, help="Cache directory to create or resume")
    p.add_argument("--splits", nargs="+", default=["tr", "cv", "tt"], choices=["tr", "cv", "tt"])
    p.add_argument("--batch_size", type=int, default=64, help="GPU inference batch size")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers for npz loading")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                   help="Storage dtype. float32 is the default and numerically closest to online inference.")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")
    p.add_argument("--verify_only", action="store_true", help="Only check the existing cache")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))

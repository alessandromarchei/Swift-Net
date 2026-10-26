#!/usr/bin/env python3
"""Paper-style evaluation of a SwiftNet checkpoint (any config) on the test split.

Usage:

    python scripts/eval_swiftnet.py --conf_dir configs/lrs2_SwiftNet_6.yml \
        --ckpt_path Experiments/LRS2_SwiftNet_6/checkpoints/epoch-166.ckpt

Metrics, per target utterance (AV-TSE: one sample per target speaker of each mixture) and averaged:

    si_snr / si_snr_i   scale-invariant SNR (= SI-SDR) and its improvement over the mixture
    snr    / snr_i      plain SNR. This is what look2hear's ALLMetricsTracker calls "sdr"
    sdr    / sdr_i      BSS-eval v3 SDR (512-tap distortion filter, same definition as mir_eval
                        bss_eval_sources), the "SDR" usually reported in AVSS/AV-TSE papers
    pesq, stoi          optional (--no_pesq_stoi to skip)
    test_loss           -mean(si_snr) with the val loss of the config, i.e. the quantity logged as
                        test_loss by test_lightning_checkpoint.py and val_loss during training

The checkpoint is either a Lightning .ckpt (audio_model.* / video_model.* keys) or an exported
best_model.pth. The visual input follows the config (cached embeddings or online FRCNN backbone);
--visual overrides it.

Outputs, in <out_root>/<exp_name>[-N]/:

    summary.txt        all statistics (mean, std, median) and the evaluation setup
    metrics.json       the same numbers, machine readable
    per_sample.csv     metrics of every evaluated sample
    config.yml         the config used
    examples/NN_<id>/  mixture.wav, target_gt.wav, estimate.wav, compare.wav
                       (mixture | gt | estimate separated by silence), clipped to [-1, 1]
    estimates/         the same N estimates side by side, for quick listening
"""
import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import soundfile as sf
import torch
import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.audio import signal_distortion_ratio

import look2hear.datas
import look2hear.losses
import look2hear.models
import look2hear.videomodels
from look2hear.losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr
from look2hear.utils.checkpoint_utils import check_architecture

SPLIT_DIRS = {"test": "test_dir", "val": "valid_dir", "train": "train_dir"}
METRICS = ["si_snr", "si_snr_i", "snr", "snr_i", "sdr", "sdr_i", "pesq", "stoi"]


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def load_checkpoint(path):
    """Return (audio_state_dict, video_state_dict or None, raw checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt and any(k.startswith("audio_model.") for k in ckpt["state_dict"]):
        sd = ckpt["state_dict"]
        # torch.compile wrappers prefix keys with _orig_mod.
        sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        audio = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
        video = {k[len("video_model."):]: v for k, v in sd.items() if k.startswith("video_model.")}
        return audio, (video or None), ckpt
    if "model_name" in ckpt and "state_dict" in ckpt:
        # best_model.pth from audio_train.py (export_payload).
        video = ckpt["visual_encoder"]["state_dict"] if "visual_encoder" in ckpt else None
        return ckpt["state_dict"], video, ckpt
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def build_models(conf, ckpt_path, visual_mode, device):
    audio_sd, video_sd, ckpt = load_checkpoint(ckpt_path)

    audio = getattr(look2hear.models, conf["audionet"]["audionet_name"])(
        sample_rate=conf["datamodule"]["data_config"]["sample_rate"],
        **copy.deepcopy(conf["audionet"]["audionet_config"]),
    )
    if hasattr(audio, "architecture_summary"):
        check_architecture(ckpt, audio.architecture_summary())
    audio.load_state_dict(audio_sd, strict=True)

    notes = []
    video = None
    if visual_mode == "online":
        video = getattr(look2hear.videomodels, conf["videonet"]["videonet_name"])(
            **conf["videonet"]["videonet_config"]
        )
        if video_sd is not None:
            video.load_state_dict(video_sd, strict=True)
            notes.append("visual encoder: weights from the checkpoint")
        else:
            # Checkpoint trained on cached embeddings: they come from the frozen pretrained backbone.
            notes.append("visual encoder: pretrained backbone from the config (checkpoint has no video weights)")
        if device.type == "cpu":
            # Measured: FRCNN embeddings on CPU differ by ~60% from the GPU ones used in training and in the
            # cache, and SI-SNRi drops by several dB. Evaluate online checkpoints on the GPU.
            notes.append("WARNING: online visual encoder on CPU does not match the GPU embeddings of training")
    else:
        if ckpt.get("visual_encoder_trainable", False) or (
            "visual_encoder" in ckpt and ckpt.get("model_name") is not None
        ):
            raise ValueError(
                "The checkpoint fine-tuned the visual encoder, but cached embeddings come from the pretrained "
                "backbone. Use --visual online."
            )
        notes.append("visual input: cached embeddings " + conf["datamodule"]["data_config"]["visual_features_path"])

    audio.to(device).eval()
    if video is not None:
        video.to(device).eval()
    info = {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "n_params_audio": sum(p.numel() for p in audio.parameters()),
        "n_params_video": sum(p.numel() for p in video.parameters()) if video is not None else None,
        "notes": notes,
    }
    return audio, video, info


@torch.no_grad()
def separate(audio, video, visual_mode, mix, mouth):
    """Same forward as AudioVisualLightningModule.forward."""
    if visual_mode == "cached":
        mouth_emb = mouth.type_as(mix)
    else:
        if mouth.ndim == 4:
            mouth = mouth.unsqueeze(1)
        mouth_emb = video(mouth.type_as(mix))
    return audio(mix, mouth_emb)


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
_PIT_SISNR = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
_PIT_SNR = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")


def bss_sdr(est, ref, filter_length):
    """BSS-eval v3 SDR per source, float64 on CPU. est, ref: [n_src, T]."""
    return signal_distortion_ratio(est.double().cpu(), ref.double().cpu(), filter_length=filter_length).float()


def compute_metrics(mix, ref, est, sr, filter_length, with_pesq_stoi):
    """mix [T], ref [n_src, T], est [n_src, T]. Returns a list of dicts, one per source."""
    n_src = ref.shape[0]
    mix_rep = mix.unsqueeze(0).expand(n_src, -1)
    # Permutation chosen by SI-SNR (trivial for TSE, n_src=1), then every metric uses it.
    _, est = _PIT_SISNR(est[None], ref[None], return_ests=True)
    est = est[0]

    def per_src(loss_fn, e):
        return -torch.diagonal(loss_fn(e[None], ref[None])[0]).cpu()

    si_snr = per_src(pairwise_neg_sisdr, est)
    si_snr_mix = per_src(pairwise_neg_sisdr, mix_rep)
    snr = per_src(pairwise_neg_snr, est)
    snr_mix = per_src(pairwise_neg_snr, mix_rep)
    sdr = bss_sdr(est, ref, filter_length)
    sdr_mix = bss_sdr(mix_rep, ref, filter_length)

    rows = []
    for s in range(n_src):
        row = {
            "si_snr": si_snr[s].item(),
            "si_snr_i": (si_snr[s] - si_snr_mix[s]).item(),
            "snr": snr[s].item(),
            "snr_i": (snr[s] - snr_mix[s]).item(),
            "sdr": sdr[s].item(),
            "sdr_i": (sdr[s] - sdr_mix[s]).item(),
            "pesq": float("nan"),
            "stoi": float("nan"),
        }
        if with_pesq_stoi:
            from pypesq import pesq
            from pystoi import stoi

            r, e = ref[s].cpu().numpy(), est[s].cpu().numpy()
            try:
                row["pesq"] = float(pesq(r, e, sr))
            except Exception:
                pass
            row["stoi"] = float(stoi(r, e, sr, extended=False))
        rows.append(row)
    return rows, est


# ----------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------
def unique_dir(root, name):
    os.makedirs(root, exist_ok=True)
    path, i = os.path.join(root, name), 0
    while os.path.exists(path):
        i += 1
        path = os.path.join(root, f"{name}-{i}")
    os.makedirs(path)
    return path


def to_wav(x):
    """Listenable waveform: clipped (saturated) to [-1, 1]. Returns (array, number of clipped samples)."""
    x = np.asarray(x, dtype=np.float32)
    clipped = int(np.sum(np.abs(x) > 1.0))
    return np.clip(x, -1.0, 1.0), clipped


def save_example(out_dir, name, mix, ref, est, sr):
    d = os.path.join(out_dir, "examples", name)
    os.makedirs(d, exist_ok=True)
    mix_w, mix_c = to_wav(mix)
    ref_w, ref_c = to_wav(ref)
    est_w, est_c = to_wav(est)
    sf.write(os.path.join(d, "mixture.wav"), mix_w, sr, subtype="PCM_16")
    sf.write(os.path.join(d, "target_gt.wav"), ref_w, sr, subtype="PCM_16")
    sf.write(os.path.join(d, "estimate.wav"), est_w, sr, subtype="PCM_16")
    gap = np.zeros(int(0.5 * sr), dtype=np.float32)
    sf.write(
        os.path.join(d, "compare.wav"),
        np.concatenate([mix_w, gap, ref_w, gap, est_w]),
        sr,
        subtype="PCM_16",
    )
    sf.write(os.path.join(out_dir, "estimates", f"{name}.wav"), est_w, sr, subtype="PCM_16")
    return {"mixture": mix_c, "target_gt": ref_c, "estimate": est_c}


def stats(values):
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return None
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def write_summary(path, setup, summary, examples):
    lines = ["SwiftNet evaluation", "=" * 70]
    for k, v in setup.items():
        if isinstance(v, list):
            for item in v:
                lines.append(f"{k:<22}{item}")
        else:
            lines.append(f"{k:<22}{v}")
    lines += ["", "Paper-style results (dB unless noted, mean over target utterances)", "-" * 70]
    for name in ["si_snr_i", "sdr_i", "snr_i", "si_snr", "sdr", "snr", "pesq", "stoi"]:
        s = summary["metrics"].get(name)
        if s is None:
            continue
        lines.append(f"{name.upper():<10} {s['mean']:8.3f}   (std {s['std']:6.3f}, median {s['median']:7.3f}, "
                     f"min {s['min']:7.3f}, max {s['max']:7.3f}, n={s['n']})")
    lines += [
        "",
        f"test_loss  {summary['test_loss']:8.4f}   (= -mean SI-SNR, the val/test loss of training)",
        f"mixture    SI-SNR {summary['mixture']['si_snr']:7.3f}   SNR {summary['mixture']['snr']:7.3f}   "
        f"SDR {summary['mixture']['sdr']:7.3f}",
        "",
        "Definitions",
        "-" * 70,
        "SI-SNR   scale-invariant SNR on zero-mean signals (= SI-SDR). X_i = X(est) - X(mixture).",
        "SNR      10 log10(|s|^2 / |s - s_hat|^2) on zero-mean signals; look2hear calls it 'sdr'.",
        "SDR      BSS-eval v3 SDR with a 512-tap distortion filter (mir_eval bss_eval_sources);",
        "         this is the SDR(i) reported in AVSS/AV-TSE papers (CTCNet, RTFS-Net, Swift-Net).",
        "PESQ     pypesq (narrow-band ITU-T P.862), same package as look2hear. STOI: pystoi.",
        "",
        f"Inference time: {summary['time_per_sample_ms']:.1f} ms/sample, real-time factor {summary['rtf']:.4f}",
    ]
    if examples:
        lines += ["", "Saved examples (examples/<name>/{mixture,target_gt,estimate,compare}.wav)", "-" * 70]
        for ex in examples:
            clip = ex["clipped_samples"]
            lines.append(f"{ex['name']:<48} SI-SNRi {ex['si_snr_i']:6.2f}  SDRi {ex['sdr_i']:6.2f}  "
                         f"clipped est={clip['estimate']} gt={clip['target_gt']} mix={clip['mixture']}")
        lines.append("compare.wav = mixture | 0.5 s silence | ground truth | 0.5 s silence | estimate")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(args):
    with open(args.conf_dir) as f:
        conf = yaml.safe_load(f)
    data_cfg = conf["datamodule"]["data_config"]

    if args.segment == "full":
        data_cfg["segment"] = None  # whole utterances, paper protocol
    elif args.segment != "config":
        data_cfg["segment"] = float(args.segment)
    if args.visual == "online":
        data_cfg["visual_features_path"] = None
    elif args.visual == "cached" and not data_cfg.get("visual_features_path"):
        raise ValueError("--visual cached requires datamodule.data_config.visual_features_path in the config")
    visual_mode = "cached" if data_cfg.get("visual_features_path") else "online"

    device = torch.device(args.device)
    sr = data_cfg["sample_rate"]

    datamodule = getattr(look2hear.datas, conf["datamodule"]["data_name"])(**data_cfg)
    datamodule.setup()
    dataset = dict(zip(["train", "val", "test"], datamodule.make_sets))[args.split]
    n_items = len(dataset.mix)  # __len__ can overcount when short utterances are dropped
    if args.max_samples:
        n_items = min(n_items, args.max_samples)
    if visual_mode == "cached":
        max_sec = dataset.visual_cache.metadata["segment_duration"]
        longest = max(m[1] for m in dataset.mix[:n_items]) / sr
        if data_cfg["segment"] is None and longest > max_sec + 1e-6:
            raise ValueError(f"Cached embeddings cover {max_sec}s but utterances last up to {longest:.2f}s. "
                             "Use --visual online or --segment config.")

    audio, video, model_info = build_models(conf, args.ckpt_path, visual_mode, device)

    ckpt_stem = Path(args.ckpt_path).stem
    exp_name = args.exp_name or f"{Path(args.conf_dir).stem}__{Path(args.ckpt_path).parent.parent.name}__{ckpt_stem}__{args.split}"
    out_dir = unique_dir(args.out_root, exp_name)
    os.makedirs(os.path.join(out_dir, "estimates"), exist_ok=True)
    with open(os.path.join(out_dir, "config.yml"), "w") as f:
        yaml.safe_dump(conf, f, sort_keys=False)

    rng = random.Random(args.seed)
    example_idx = set(rng.sample(range(n_items), min(args.num_audio, n_items)))

    loader = DataLoader(Subset(dataset, range(n_items)), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    per_sample, examples = [], []
    mix_acc = {"si_snr": [], "snr": [], "sdr": []}
    infer_time, audio_sec = 0.0, 0.0
    csv_path = os.path.join(out_dir, "per_sample.csv")
    with open(csv_path, "w", newline="") as fcsv, Progress(
        TextColumn("[bold blue]Evaluating"), BarColumn(bar_width=None), MofNCompleteColumn(), "•",
        TimeRemainingColumn(), "•", TextColumn("{task.fields[running]}"),
    ) as progress:
        writer = csv.DictWriter(fcsv, fieldnames=["idx", "mixture", "target"] + METRICS)
        writer.writeheader()
        task = progress.add_task("eval", total=n_items, running="")
        for idx, (mix, ref, mouth, key) in enumerate(loader):
            mix, ref, mouth = mix.to(device), ref.to(device), mouth.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            est = separate(audio, video, visual_mode, mix, mouth)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time += time.perf_counter() - t0
            audio_sec += mix.shape[-1] / sr

            ref = ref if ref.ndim == 3 else ref.unsqueeze(1)  # [1, n_src, T]
            est = est[..., : ref.shape[-1]]
            rows, est_aligned = compute_metrics(mix[0], ref[0], est[0], sr, args.sdr_filter_length,
                                                not args.no_pesq_stoi)

            # sources[idx] is [wav, mouth, len] for n_src=1, a list of them for n_src=2.
            src_paths = [dataset.sources[idx]] if dataset.n_src == 1 else dataset.sources[idx]
            for s, row in enumerate(rows):
                target_id = Path(src_paths[s][0]).parent.name  # s1 / s2
                writer.writerow({"idx": idx, "mixture": key[0], "target": target_id, **row})
                per_sample.append(row)
                if idx in example_idx:
                    name = f"{len(examples):02d}_{Path(key[0]).stem[:32]}_{target_id}"
                    clipped = save_example(out_dir, name, mix[0].cpu().numpy(), ref[0, s].cpu().numpy(),
                                           est_aligned[s].cpu().numpy(), sr)
                    examples.append({"name": name, "idx": idx, "mixture": key[0], "target": target_id,
                                     "si_snr_i": row["si_snr_i"], "sdr_i": row["sdr_i"],
                                     "clipped_samples": clipped})
            mix_acc["si_snr"] += [r["si_snr"] - r["si_snr_i"] for r in rows]
            mix_acc["snr"] += [r["snr"] - r["snr_i"] for r in rows]
            mix_acc["sdr"] += [r["sdr"] - r["sdr_i"] for r in rows]

            if idx % 20 == 0:
                m = lambda k: np.nanmean([r[k] for r in per_sample])
                progress.update(task, running=f"SI-SNRi {m('si_snr_i'):.2f}  SDRi {m('sdr_i'):.2f}")
            progress.advance(task)

    summary = {
        "metrics": {k: stats([r[k] for r in per_sample]) for k in METRICS},
        "mixture": {k: float(np.mean(v)) for k, v in mix_acc.items()},
        "test_loss": -float(np.mean([r["si_snr"] for r in per_sample])),
        "time_per_sample_ms": 1000 * infer_time / max(n_items, 1),
        "rtf": infer_time / max(audio_sec, 1e-9),
    }
    setup = {
        "config": os.path.abspath(args.conf_dir),
        "checkpoint": os.path.abspath(args.ckpt_path),
        "checkpoint epoch": model_info["epoch"],
        "model": conf["audionet"]["audionet_name"],
        "params (audio)": f"{model_info['n_params_audio'] / 1e6:.3f} M",
        "split": f"{args.split} ({data_cfg[SPLIT_DIRS[args.split]]})",
        "samples": f"{len(per_sample)} target utterances from {n_items} dataset items"
                   + (f" (limited by --max_samples)" if args.max_samples else ""),
        "segment": "full utterance" if data_cfg["segment"] is None else f"{data_cfg['segment']} s crop",
        "visual mode": visual_mode,
        "device": str(device),
        "notes": model_info["notes"],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"setup": setup, "summary": summary, "examples": examples}, f, indent=2, default=str)
    text = write_summary(os.path.join(out_dir, "summary.txt"), setup, summary, examples)
    print(text)
    print(f"\nResults written to {out_dir}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conf_dir", required=True, help="YAML config used to train the checkpoint")
    p.add_argument("--ckpt_path", required=True, help="Lightning .ckpt or exported best_model.pth")
    p.add_argument("--split", default="test", choices=list(SPLIT_DIRS))
    p.add_argument("--segment", default="full",
                   help="'full' (whole utterances, paper protocol), 'config' (crop of the config, as in training) "
                        "or a length in seconds")
    p.add_argument("--visual", default="auto", choices=["auto", "online", "cached"],
                   help="auto: as in the config. online: FRCNN backbone on mouth frames. cached: precomputed embeddings")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_audio", type=int, default=10, help="number of examples saved as audio")
    p.add_argument("--seed", type=int, default=0, help="seed for the choice of the saved examples")
    p.add_argument("--max_samples", type=int, default=None, help="evaluate only the first N dataset items")
    p.add_argument("--sdr_filter_length", type=int, default=512, help="BSS-eval distortion filter length")
    p.add_argument("--no_pesq_stoi", action="store_true", help="skip PESQ and STOI (faster)")
    p.add_argument("--out_root", default=os.path.join("Experiments", "eval"))
    p.add_argument("--exp_name", default=None,
                   help="name of the eval folder. Default: <config>__<run>__<ckpt>__<split>")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))

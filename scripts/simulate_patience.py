#!/usr/bin/env python
"""
A-posteriori simulation of ReduceLROnPlateau with lower patience on the FIRST
LR phase of finished W&B runs.

For every run it downloads `val_pit_sisnr` (per epoch) and `learning_rate`,
cuts the curve at the first real LR drop (phase 1, highest LR), then replays
the plateau logic (mode=min on val_loss = -sisnr, rel. threshold like PyTorch)
for each patience value and reports where the first LR drop would have
happened and which best SI-SNR you would have had at that point.

Caveat: after the simulated drop the real trajectory would differ; this only
tells you how much of phase 1 was "wasted" waiting for patience.

Examples:
  # list runs in the project
  python scripts/simulate_patience.py --entity ENT --list
  # simulate patience 2..10 on two runs (names or ids), save plots
  python scripts/simulate_patience.py --entity ENT \
      --runs LRS2_SwiftNet_6_freqmix_mhsa_unfold_v3-2 LRS2_SwiftNet_6 \
      --patience 2 3 4 5 6 8 10 --out results/patience_sim
  # all runs whose name matches a regex, show interactively
  python scripts/simulate_patience.py --entity ENT --filter "freqmix" --show
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


def fetch_run(run, cache_dir=None):
    path = os.path.join(cache_dir, f"{run.id}.csv") if cache_dir else None
    if path and os.path.exists(path):
        return pd.read_csv(path)
    sis = pd.DataFrame(run.scan_history(keys=["val_pit_sisnr", "epoch"]))
    lr = pd.DataFrame(run.scan_history(keys=["learning_rate", "epoch"]))
    if sis.empty:
        return sis
    sis = sis.groupby("epoch", as_index=False)["val_pit_sisnr"].last()
    if not lr.empty:
        lr = lr.groupby("epoch", as_index=False)["learning_rate"].last()
        sis = sis.merge(lr, on="epoch", how="left")
    else:
        sis["learning_rate"] = np.nan
    sis = sis.sort_values("epoch").reset_index(drop=True)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        sis.to_csv(path, index=False)
    return sis


def first_phase(df, max_epoch=None):
    """Rows of phase 1 (before the first LR decrease). Fallback: max_epoch or all."""
    lr = df["learning_rate"].values
    if not np.all(np.isnan(lr)):
        lr0 = lr[~np.isnan(lr)][0]
        drop = np.where(lr < lr0 * 0.999)[0]
        if len(drop):
            return df.iloc[: drop[0]].reset_index(drop=True), int(df["epoch"].iloc[drop[0]])
    if max_epoch is not None:
        return df[df["epoch"] <= max_epoch].reset_index(drop=True), None
    return df, None


def simulate(sisnr, patience, threshold=1e-4, mode="rel"):
    """Replay ReduceLROnPlateau(mode='min') on loss=-sisnr.
    Returns (index of the epoch where LR would be reduced, best sisnr then) or (None, best)."""
    best = np.inf
    bad = 0
    for i, s in enumerate(sisnr):
        loss = -s
        if mode == "rel":  # identical to torch: a < best * (1 - threshold)
            better = loss < best * (1 - threshold)
        else:
            better = loss < best - threshold
        if better:
            best, bad = loss, 0
        else:
            bad += 1
        if bad > patience:
            return i, -best
    return None, -best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=None, help="W&B entity (default: your default entity)")
    p.add_argument("--project", default="SwiftNet")
    p.add_argument("--runs", nargs="*", default=[], help="run names or ids")
    p.add_argument("--filter", default=None, help="regex on run name")
    p.add_argument("--list", action="store_true", help="list runs and exit")
    p.add_argument("--patience", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10])
    p.add_argument("--threshold", type=float, default=1e-4, help="ReduceLROnPlateau threshold")
    p.add_argument("--threshold-mode", choices=["rel", "abs"], default="rel")
    p.add_argument("--max-epoch", type=int, default=None,
                   help="end of phase 1 if learning_rate was not logged")
    p.add_argument("--full", action="store_true", help="plot the whole curve, not only phase 1")
    p.add_argument("--cache", default=None, help="dir to cache downloaded histories as csv")
    p.add_argument("--out", default=None, help="dir to save png/csv")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()

    api = wandb.Api()
    prefix = f"{a.entity}/{a.project}" if a.entity else a.project
    all_runs = list(api.runs(prefix))

    if a.list:
        for r in all_runs:
            print(f"{r.id:12s} {r.state:9s} {r.name}")
        return

    sel = [r for r in all_runs if r.name in a.runs or r.id in a.runs]
    if a.filter:
        sel += [r for r in all_runs if re.search(a.filter, r.name) and r not in sel]
    if not sel:
        raise SystemExit("No runs selected (use --runs/--filter, --list to see them).")

    rows = []
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(a.patience)))
    for run in sel:
        df = fetch_run(run, a.cache)
        if not df.empty:
            df = df.dropna(subset=["val_pit_sisnr"]).reset_index(drop=True)
        if df.empty:
            print(f"[skip] {run.name}: no val_pit_sisnr")
            continue
        ph, real_drop = first_phase(df, a.max_epoch)
        ep, s = ph["epoch"].values, ph["val_pit_sisnr"].values
        best_phase = s.max()
        print(f"\n=== {run.name} ({run.id})  phase-1 epochs: {ep[0]}..{ep[-1]}"
              f"  real 1st LR drop @ epoch {real_drop}  best phase-1 SI-SNR {best_phase:.3f}")
        print(f"{'patience':>8} {'drop@ep':>8} {'best@drop':>10} {'Δ vs full':>10} {'epochs saved':>13}")

        fig, ax = plt.subplots(figsize=(11, 5))
        plot_df = df if a.full else ph
        ax.plot(plot_df["epoch"], plot_df["val_pit_sisnr"], color="#d62839", lw=1.5, label="val_pit_sisnr")
        ax.plot(ph["epoch"], np.maximum.accumulate(s), color="gray", ls=":", lw=1, label="running best")
        if real_drop is not None:
            ax.axvline(real_drop, color="black", lw=1.5, label=f"real drop (ep {real_drop})")
        for c, pat in zip(colors, a.patience):
            idx, best = simulate(s, pat, a.threshold, a.threshold_mode)
            if idx is None:
                drop_ep, saved = None, 0
                print(f"{pat:>8} {'none':>8} {best:>10.3f} {best - best_phase:>10.3f} {0:>13}")
            else:
                drop_ep = int(ep[idx])
                saved = int(ep[-1] - drop_ep)
                print(f"{pat:>8} {drop_ep:>8} {best:>10.3f} {best - best_phase:>10.3f} {saved:>13}")
                ax.axvline(drop_ep, color=c, ls="--", lw=1, label=f"p={pat}: ep {drop_ep}, {best:.2f} dB")
            rows.append(dict(run=run.name, patience=pat, drop_epoch=drop_ep, best_at_drop=best,
                             best_phase1=best_phase, delta=best - best_phase, epochs_saved=saved,
                             real_drop=real_drop))
        ax.set_xlabel("epoch"); ax.set_ylabel("val SI-SNR (dB)")
        ax.set_title(f"{run.name} — simulated first LR drop vs patience")
        ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            fig.savefig(os.path.join(a.out, f"{run.name}_patience.png"), dpi=130)

    if not rows:
        return
    res = pd.DataFrame(rows)
    # summary: SI-SNR lost vs epochs saved, one line per run
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, g in res.groupby("run"):
        a1.plot(g["patience"], g["delta"], marker="o", label=name)
        a2.plot(g["patience"], g["epochs_saved"], marker="o", label=name)
    a1.set_xlabel("patience"); a1.set_ylabel("best@drop − best phase-1 (dB)"); a1.set_title("SI-SNR lost at 1st drop")
    a2.set_xlabel("patience"); a2.set_ylabel("epochs"); a2.set_title("phase-1 epochs saved")
    for x in (a1, a2):
        x.grid(alpha=.3)
    a2.legend(fontsize=7)
    fig.tight_layout()
    if a.out:
        fig.savefig(os.path.join(a.out, "summary.png"), dpi=130)
        res.to_csv(os.path.join(a.out, "summary.csv"), index=False)
        print(f"\nSaved plots/csv to {a.out}")
    if a.show or not a.out:
        plt.show()


if __name__ == "__main__":
    main()

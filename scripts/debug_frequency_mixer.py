"""Diagnostics for the convolutional frequency mixer (SUM vs CONCAT) in SwiftNet's FTGS block.

Runs on CPU with the real repository classes (SwiftNet, FTGS, DilatedFrequencyMixer) and
SYNTHETIC inputs of realistic shape. No dataset is read and no training is run.

    python scripts/debug_frequency_mixer.py --config configs/lrs2_SwiftNet_6_freqmix_concat_v1.yml

Sections:
    1. Resolved configuration and instantiated modules (YAML -> SwiftNet -> FTGS -> mixer).
    2. SUM vs CONCAT on identical inputs and shared weights: parameters, output statistics,
       and manual recomputation of the fusion path via hooks.
    3. Forward/backward/optimizer step on the full SwiftNet from --config: gradients,
       parameter updates, optimizer membership, and how many times FTGS calls the mixer.
    4. Gradient accumulation arithmetic (batch 8 x1 vs batch 4 x2) on a tiny synthetic task.
"""

import argparse
import copy
import inspect
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from look2hear.models import SwiftNet  # noqa: E402
from look2hear.models.SwiftNet import FTGS  # noqa: E402
from look2hear.models.frequency_mixer import (  # noqa: E402
    DilatedFrequencyMixer,
    describe_frequency_module,
)


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_swiftnet(conf, fusion=None):
    """Build the real SwiftNet from the YAML. Optionally override branch_fusion (SUM/CONCAT comparison)."""
    ac = copy.deepcopy(conf["audionet"]["audionet_config"])
    if fusion is not None:
        ac["audio_params"]["frequency_mixer"]["conv"]["branch_fusion"] = fusion
    sample_rate = conf["datamodule"]["data_config"]["sample_rate"]
    return SwiftNet(sample_rate=sample_rate, **ac)


def first_ftgs(model):
    nets = model.separator.audio_net
    return nets[0] if isinstance(nets, nn.ModuleList) else nets


def mixer_of(model):
    return first_ftgs(model).process_f


def count_params(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# 1. Resolved configuration
# ---------------------------------------------------------------------------
def section_config(conf, model):
    header("1. Resolved configuration and instantiated modules")
    ac = conf["audionet"]["audionet_config"]
    yaml_mixer = ac["audio_params"].get("frequency_mixer")
    print("YAML audio_params.frequency_mixer:", yaml_mixer)

    ftgs = first_ftgs(model)
    print("FTGS class:", type(ftgs).__name__, "| frequency_mixer_type:", ftgs.frequency_mixer_type)
    print("FTGS.frequency_mixer_cfg == YAML:", ftgs.frequency_mixer_cfg == yaml_mixer)

    info = describe_frequency_module(mixer_of(model))
    for key, value in info.items():
        print(f"  {key}: {value}")

    nets = model.separator.audio_net
    n_ftgs = len(nets) if isinstance(nets, nn.ModuleList) else 1
    repeats = model.separator.fusion_repeats + model.separator.audio_repeats
    print(f"FTGS module instances: {n_ftgs} | FTGS repetitions per forward: {repeats}")
    print(f"audio_shared: {model.separator.audio_shared} (True = one FTGS reused {repeats} times)")

    # Check that no original GRNN (DualPathRNN) hides in the frequency slot and that the mixer is not an
    # unused leftover: names under process_f must all belong to the mixer.
    names = [n for n, _ in mixer_of(model).named_modules()]
    print("process_f submodules (first 12):", names[:12])
    print("process_f contains DualPathRNN:", any(type(m).__name__ == "DualPathRNN" for m in mixer_of(model).modules()))

    # Source-level check: FTGS.forward must not detach or disable grads.
    src = inspect.getsource(FTGS.forward) + inspect.getsource(DilatedFrequencyMixer.forward)
    print("FTGS/mixer forward contains detach/no_grad:", "detach" in src or "no_grad" in src)

    # Each FTGS repetition must call process_f exactly once.
    calls = {"n": 0}
    handle = mixer_of(model).register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 32000), torch.randn(1, 512, 50))
    model.train()
    handle.remove()
    print(f"process_f forward calls in one model forward: {calls['n']} (expected {repeats} "
          f"if shared, {repeats} modules if unshared)")
    return info


# ---------------------------------------------------------------------------
# 2. SUM vs CONCAT
# ---------------------------------------------------------------------------
def make_mixer(fusion, seed=0):
    torch.manual_seed(seed)
    return DilatedFrequencyMixer(
        channels=64,
        kernel_sizes=[3, 3, 3, 3, 3],
        dilations=[1, 2, 4, 8, 16],
        branch_fusion=fusion,
        hidden_channels=64,
        depthwise=True,
        pointwise=True,
        activation="prelu",
        normalization="layernorm",
        residual=True,
        dropout=0.0,
        num_blocks=1,
    ).eval()


def stats(t):
    return f"mean={t.mean().item():+.6f} std={t.std().item():.6f}"


def section_sum_vs_concat():
    header("2. SUM vs CONCAT on identical inputs and shared weights (synthetic, real DilatedFrequencyMixer)")
    sum_m = make_mixer("sum")
    cat_m = make_mixer("concat")

    # Copy every weight that exists in both, so the only difference is the fusion path.
    missing, unexpected = cat_m.load_state_dict(sum_m.state_dict(), strict=False)
    print("CONCAT-only keys (missing when loading SUM weights):", missing)
    print("unexpected keys:", unexpected)

    print(f"params SUM={count_params(sum_m)} CONCAT={count_params(cat_m)} "
          f"diff={count_params(cat_m) - count_params(sum_m)}")
    print("SUM fuse module:", type(sum_m.blocks[0].fuse).__name__,
          "| CONCAT fuse module:", type(cat_m.blocks[0].fuse).__name__,
          tuple(cat_m.blocks[0].fuse.weight.shape))

    torch.manual_seed(1)
    x = torch.randn(2, 64, 50, 65)  # [B, C, T, F]
    with torch.no_grad():
        y_sum = sum_m(x)
        y_cat = cat_m(x)

    diff = (y_sum - y_cat)
    cos = F.cosine_similarity(y_sum.flatten(), y_cat.flatten(), dim=0).item()
    print("SUM   output:", stats(y_sum))
    print("CONCAT output:", stats(y_cat))
    print(f"max|diff|={diff.abs().max().item():.6f} mean|diff|={diff.abs().mean().item():.6f} cos={cos:.6f}")
    print("input shape:", tuple(x.shape), "| output shape:", tuple(y_sum.shape))

    # Manual recomputation of each fusion path, using the branch outputs captured by hooks.
    for name, m in (("SUM", sum_m), ("CONCAT", cat_m)):
        block = m.blocks[0]
        outs = {}
        hooks = [b.register_forward_hook(lambda mod, i, o, k=k: outs.__setitem__(k, o)) for k, b in enumerate(block.branches)]
        with torch.no_grad():
            y = m(x)
        for h in hooks:
            h.remove()
        with torch.no_grad():
            ordered = [outs[k] for k in range(len(block.branches))]
            if name == "SUM":
                fused = torch.stack(ordered).sum(0)  # element-wise sum over branches, no scaling
            else:
                fused = F.conv2d(torch.cat(ordered, dim=1), block.fuse.weight, block.fuse.bias)  # channel concat -> 1x1 projection
            manual = block.project(block.act_mid(fused)) + x
        err = (manual - y).abs().max().item()
        print(f"{name}: recomputed fusion path vs forward, max abs error = {err:.2e} "
              f"(branches captured: {len(ordered)}, branch_fusion attr = {block.branch_fusion})")
        assert err < 1e-5, f"{name} forward does not match its documented computation"

    # Every branch must influence the output: add random noise to one branch output at a time.
    # (A constant offset is not used: for SUM it shifts the same pre-activation whichever branch carries it,
    # so every branch would show an identical effect by construction.)
    for name, m in (("SUM", sum_m), ("CONCAT", cat_m)):
        block = m.blocks[0]
        with torch.no_grad():
            base = m(x)
        effects = []
        for k, branch in enumerate(block.branches):
            gen = torch.Generator().manual_seed(100 + k)
            h = branch.register_forward_hook(lambda mod, i, o, g=gen: o + 0.5 * torch.randn(o.shape, generator=g))
            with torch.no_grad():
                pert = m(x)
            h.remove()
            effects.append((pert - base).abs().max().item())
        print(f"{name}: max output change when perturbing each branch: " + ", ".join(f"b{k}={e:.4f}" for k, e in enumerate(effects)))
        assert all(e > 0 for e in effects), f"{name}: a branch does not reach the output"

    # CONCAT must depend on channel order of the branches (it is a learned projection of the concatenation).
    with torch.no_grad():
        block = cat_m.blocks[0]
        y_in = block.act_in(block.expand(block.norm(x)))
        ordered = [b(y_in) for b in block.branches]
        a = F.conv2d(torch.cat(ordered, 1), block.fuse.weight, block.fuse.bias)
        bshuf = F.conv2d(torch.cat(ordered[::-1], 1), block.fuse.weight, block.fuse.bias)
    print(f"CONCAT: reversing branch order changes fused tensor by max {(a - bshuf).abs().max().item():.4f} "
          "(expected > 0: fuse weights are position-specific)")

    return sum_m, cat_m


# ---------------------------------------------------------------------------
# 3. Forward / backward / optimizer step on the full model
# ---------------------------------------------------------------------------
def section_training_step(conf, model):
    header("3. Forward/backward/optimizer step on the full SwiftNet (synthetic data, CPU)")
    torch.manual_seed(2)
    model.train()
    audio = torch.randn(2, 32000) * 0.1  # 2 s at 16 kHz
    emb = torch.randn(2, 512, 50)        # 2 s of visual features at 25 fps
    target = torch.randn(2, 1, 32000) * 0.1

    mixer = mixer_of(model)
    mixer_params = dict(mixer.named_parameters())
    before = {n: p.detach().clone() for n, p in mixer_params.items()}

    opt_cfg = conf["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    in_optimizer = all(id(p) in opt_ids for p in mixer_params.values())
    print(f"mixer parameters: {len(mixer_params)} tensors, {count_params(mixer)} elements; "
          f"all in optimizer: {in_optimizer}")

    out = model(audio, emb)
    print("model output shape:", tuple(out.shape))
    loss = F.mse_loss(out, target)
    optimizer.zero_grad()
    loss.backward()

    grads_ok = True
    print(f"{'parameter':<45} {'requires_grad':>13} {'grad_norm':>12}")
    for n, p in mixer_params.items():
        g = p.grad.norm().item() if p.grad is not None else float("nan")
        if p.grad is None or g == 0.0:
            grads_ok = False
        print(f"{n:<45} {str(p.requires_grad):>13} {g:>12.3e}")

    optimizer.step()
    print("parameter update norms (AdamW step):")
    updated_ok = True
    for n, p in mixer_params.items():
        upd = (p.detach() - before[n]).norm().item()
        if upd == 0.0:
            updated_ok = False
        print(f"  {n:<43} {upd:.3e}")

    print(f"all mixer parameters received non-zero gradients: {grads_ok}")
    print(f"all mixer parameters were updated: {updated_ok}")
    return grads_ok and updated_ok and in_optimizer


# ---------------------------------------------------------------------------
# 4. Gradient accumulation arithmetic
# ---------------------------------------------------------------------------
def section_accumulation(conf):
    header("4. Gradient accumulation: batch 4 x1 vs 2+2 accumulated (same 4 synthetic samples, same weights)")
    torch.manual_seed(3)
    model = build_swiftnet(conf).eval()  # eval: no dropout and no batch-dependent statistics
    init = copy.deepcopy(model.state_dict())
    audio = torch.randn(4, 32000) * 0.1
    emb = torch.randn(4, 512, 50)
    target = torch.randn(4, 1, 32000) * 0.1

    def grads_for(batches):
        model.load_state_dict(init)
        model.zero_grad(set_to_none=True)
        n = sum(a.shape[0] for a, _, _ in batches)
        for a, e, t in batches:
            loss = F.mse_loss(model(a, e), t, reduction="sum") / (n * t[0].numel())  # mean over all samples
            loss.backward()
        return {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}

    g_full = grads_for([(audio, emb, target)])
    g_acc = grads_for([(audio[:2], emb[:2], target[:2]), (audio[2:], emb[2:], target[2:])])
    # Global relative norm over all parameters. A per-parameter max is dominated by tiny-gradient tensors.
    num = sum(((g_full[k] - g_acc[k]) ** 2).sum() for k in g_full).sqrt()
    den = sum((g_full[k] ** 2).sum() for k in g_full).sqrt()
    worst = (num / den).item()
    print(f"global relative gradient difference (batch 4 vs 2+2 accumulated, loss averaged over samples): {worst:.2e}")
    print("Note: this emulates accumulation in plain PyTorch. Lightning's own loss scaling for "
          "accumulate_grad_batches must be checked against the installed version (not covered here).")
    return worst < 1e-4


def section_batch_independence(conf, model):
    header("5. Batch independence: does one sample's output depend on the other items in the batch?")
    model.eval()
    torch.manual_seed(11)
    audio = torch.randn(4, 32000) * 0.1
    emb = torch.randn(4, 512, 50)
    with torch.no_grad():
        batched = model(audio, emb)
        alone = torch.cat([model(audio[i : i + 1], emb[i : i + 1]) for i in range(4)])
    rel = ((batched - alone).norm() / alone.norm()).item()
    print(f"relative L2 change, batch of 4 vs each sample alone: {rel:.3e}")
    print("Expected ~1e-7 (fp32 rounding). A larger value means samples interact inside the model.")
    print("Video branch alone (LightVid, separator.video_net):")
    vnet = model.separator.video_net
    vnet = vnet[0] if isinstance(vnet, nn.ModuleList) else vnet
    with torch.no_grad():
        vb = vnet(emb)
        va = torch.cat([vnet(emb[i : i + 1]) for i in range(4)])
    print(f"  max |batch - alone| = {(vb - va).abs().max().item():.3e}")
    return rel < 1e-5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/lrs2_SwiftNet_6.yml")
    args = parser.parse_args()

    torch.manual_seed(0)
    conf = load_config(os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config)
    ac = conf["audionet"]["audionet_config"]
    fm = ac["audio_params"].get("frequency_mixer")
    if not fm or fm.get("type") != "conv":
        print("This config does not enable the conv frequency mixer; the mixer checks need type: conv.")
        return 1

    model = build_swiftnet(conf)
    results = {}
    section_config(conf, model)
    sum_m, cat_m = section_sum_vs_concat()
    results["training_step"] = section_training_step(conf, model)
    results["accumulation"] = section_accumulation(conf)
    results["batch_independence"] = section_batch_independence(conf, model)

    header("Summary")
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

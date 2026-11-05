"""Tests for the learning rate schedulers, in isolation and inside a real Lightning Trainer.

Run from the repository root:
    python -m unittest tests.test_schedulers -v
"""

import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

from look2hear.system import ConstantCosineLR, WarmupCosineLR, make_scheduler

LR = 1e-3
MIN_LR = 1e-5


def make_optimizer(lr=LR):
    return torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=lr)


def lr_curve(sched, n_steps):
    """lr used by optimizer steps 0..n_steps-1, stepping the scheduler like Lightning does."""
    lrs = []
    for _ in range(n_steps):
        lrs.append(sched.optimizer.param_groups[0]["lr"])
        sched.optimizer.step()
        sched.step()
    return lrs


class ConstantCosineLRTests(unittest.TestCase):
    def setUp(self):
        self.c, self.t = 100, 300
        self.sched = ConstantCosineLR(make_optimizer(), constant_steps=self.c, total_steps=self.t, min_lr=MIN_LR)
        self.lrs = lr_curve(self.sched, 400)

    def test_constant_phase_and_at_constant_steps(self):
        self.assertTrue(all(lr == LR for lr in self.lrs[: self.c]))
        self.assertEqual(self.lrs[self.c], LR)

    def test_cosine_phase_matches_formula_and_decreases(self):
        for step in range(self.c, self.t):
            expected = MIN_LR + (LR - MIN_LR) / 2 * (1 + math.cos(math.pi * (step - self.c) / (self.t - self.c)))
            self.assertAlmostEqual(self.lrs[step], expected, places=15)
        cosine = self.lrs[self.c : self.t + 1]
        self.assertTrue(all(a > b for a, b in zip(cosine, cosine[1:])))
        # Smooth: no jump larger than the steepest slope of the cosine, pi/2 * (LR - MIN_LR) / (T - C)
        max_jump = math.pi / 2 * (LR - MIN_LR) / (self.t - self.c)
        self.assertTrue(all(a - b <= max_jump + 1e-15 for a, b in zip(cosine, cosine[1:])))

    def test_reaches_and_keeps_min_lr(self):
        self.assertEqual(self.lrs[self.t], MIN_LR)
        self.assertTrue(all(lr == MIN_LR for lr in self.lrs[self.t :]))
        self.assertGreaterEqual(min(self.lrs), MIN_LR)

    def test_lr_follows_global_step_in_last_epoch(self):
        self.assertEqual(self.sched.last_epoch, 400)
        self.assertEqual(self.sched.get_last_lr(), [MIN_LR])

    def test_constant_steps_zero_starts_cosine_immediately(self):
        lrs = lr_curve(ConstantCosineLR(make_optimizer(), 0, 10, MIN_LR), 11)
        self.assertEqual(lrs[0], LR)
        self.assertLess(lrs[1], LR)
        self.assertEqual(lrs[10], MIN_LR)

    def test_each_param_group_anneals_from_its_own_lr(self):
        net = nn.Linear(2, 2)
        opt = torch.optim.AdamW([{"params": [net.weight]}, {"params": [net.bias], "lr": 1e-4}], lr=LR)
        sched = ConstantCosineLR(opt, 2, 6, MIN_LR)
        for _ in range(6):
            opt.step()
            sched.step()
        self.assertEqual([g["lr"] for g in opt.param_groups], [MIN_LR, MIN_LR])
        self.assertEqual(sched.base_lrs, [LR, 1e-4])

    def test_validation(self):
        for kwargs in (
            dict(constant_steps=-1, total_steps=10),
            dict(constant_steps=10, total_steps=10),
            dict(constant_steps=20, total_steps=10),
            dict(constant_steps=1.5, total_steps=10),
            dict(constant_steps="abc", total_steps=10),
            dict(constant_steps=1, total_steps=10, min_lr=-1e-6),
            dict(constant_steps=1, total_steps=10, min_lr=2 * LR),
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                ConstantCosineLR(make_optimizer(), **kwargs)
        # min_lr == initial lr is allowed (flat schedule)
        ConstantCosineLR(make_optimizer(), 0, 10, min_lr=LR)

    def test_yaml_config_values(self):
        # PyYAML reads "1e-5" and "1e5" as strings: they must still be accepted.
        cfg = yaml.safe_load(
            "scheduler:\n  sche_name: constant_cosine\n  sche_config:\n"
            "    constant_steps: 1e5\n    total_steps: 3.0e+5\n    min_lr: 1e-5\n"
        )["scheduler"]
        sched = make_scheduler(cfg["sche_name"], make_optimizer(), **cfg["sche_config"])
        self.assertIsInstance(sched, ConstantCosineLR)
        self.assertEqual((sched.constant_steps, sched.total_steps, sched.min_lr), (100000, 300000, 1e-5))
        self.assertIsInstance(make_scheduler("ConstantCosineLR", make_optimizer(), constant_steps=1, total_steps=2), ConstantCosineLR)

    def test_state_dict_resume_continues_on_curve(self):
        reference = lr_curve(ConstantCosineLR(make_optimizer(), 100, 300, MIN_LR), 300)

        opt = make_optimizer()
        sched = ConstantCosineLR(opt, 100, 300, MIN_LR)
        lr_curve(sched, 150)
        opt_state, sched_state = opt.state_dict(), sched.state_dict()

        opt2 = make_optimizer()
        sched2 = ConstantCosineLR(opt2, 100, 300, MIN_LR)  # fresh: lr would be the constant LR
        opt2.load_state_dict(opt_state)
        sched2.load_state_dict(sched_state)
        self.assertEqual(sched2.last_epoch, 150)
        self.assertEqual(lr_curve(sched2, 150), reference[150:])

    def test_checkpoint_schedule_wins_over_configured_one(self):
        sched = ConstantCosineLR(make_optimizer(), 100, 300, MIN_LR)
        lr_curve(sched, 5)
        other = ConstantCosineLR(make_optimizer(), 10, 20, 0.0)
        other.load_state_dict(sched.state_dict())
        self.assertEqual((other.constant_steps, other.total_steps, other.min_lr), (100, 300, MIN_LR))


class ExistingSchedulerTests(unittest.TestCase):
    def test_reduce_on_plateau_from_repo_config(self):
        conf = yaml.safe_load(open(REPO / "configs" / "lrs2_SwiftNet_6.yml"))["scheduler"]
        sched = make_scheduler(conf["sche_name"], make_optimizer(), **conf["sche_config"])
        self.assertIsInstance(sched, ReduceLROnPlateau)

    def test_warmup_cosine_still_built_by_name(self):
        sched = make_scheduler("WarmupCosineLR", make_optimizer(), warmup_epochs=1)
        self.assertIsInstance(sched, WarmupCosineLR)


# ----------------------------------------------------------------------
# Real Lightning Trainer with the repository's AudioVisualLightningModule
# ----------------------------------------------------------------------


class _Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16)

    def forward(self, wav, emb):
        return self.lin(wav).unsqueeze(1)


def _loss(est, target):
    return ((est - target) ** 2).mean()


def _loader(n_batches, batch_size=2):
    from torch.utils.data import DataLoader, TensorDataset

    n = n_batches * batch_size
    g = torch.Generator().manual_seed(0)
    ds = TensorDataset(torch.randn(n, 16, generator=g), torch.randn(n, 16, generator=g), torch.zeros(n, 4, 3),
                       torch.zeros(n))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class _LRRecorder:
    """Records (global_step, lr) just before every real optimizer step."""

    def __new__(cls):
        import pytorch_lightning as pl

        class Recorder(pl.Callback):
            def __init__(self):
                self.records = []

            def on_before_optimizer_step(self, trainer, pl_module, optimizer):
                self.records.append((trainer.global_step, optimizer.param_groups[0]["lr"]))

        return Recorder()


class TrainerIntegrationTests(unittest.TestCase):
    N_BATCHES = 10  # with accumulate_grad_batches=3: optimizer steps after batches 3, 6, 9 and the last one
    ACCUM = 3
    STEPS_PER_EPOCH = 4

    def _module(self, scheduler_factory, lr=LR):
        from look2hear.system import AudioVisualLightningModule

        audio = _Audio()
        opt = torch.optim.AdamW(audio.parameters(), lr=lr)
        loader = _loader(self.N_BATCHES)
        return AudioVisualLightningModule(
            audio_model=audio, video_model=None, optimizer=opt, loss_func={"train": _loss, "val": _loss},
            train_loader=loader, val_loader=_loader(1), test_loader=None,
            scheduler=scheduler_factory(opt), config={}, visual_mode="cached",
        )

    def _trainer(self, tmp, max_epochs, callbacks=()):
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint

        ckpt = ModelCheckpoint(dirpath=os.path.join(tmp, "ckpt"), monitor="val_loss", save_last=True)
        rec = _LRRecorder()
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", devices=1, logger=False,
                             callbacks=[ckpt, rec, *callbacks], enable_progress_bar=False,
                             enable_model_summary=False, num_sanity_val_steps=0, default_root_dir=tmp,
                             accumulate_grad_batches=self.ACCUM)
        return trainer, ckpt, rec

    @staticmethod
    def _expected(step, c, t):
        if step < c:
            return LR
        if step >= t:
            return MIN_LR
        return MIN_LR + (LR - MIN_LR) / 2 * (1 + math.cos(math.pi * (step - c) / (t - c)))

    def test_one_scheduler_step_per_optimizer_step_with_accumulation(self):
        c, t = 3, 9
        with tempfile.TemporaryDirectory() as tmp:
            trainer, _, rec = self._trainer(tmp, max_epochs=3)
            module = self._module(lambda opt: ConstantCosineLR(opt, c, t, MIN_LR))
            trainer.fit(module)
            n_steps = 3 * self.STEPS_PER_EPOCH
            self.assertEqual(trainer.global_step, n_steps)
            self.assertEqual([s for s, _ in rec.records], list(range(n_steps)))
            for step, lr in rec.records:
                self.assertAlmostEqual(lr, self._expected(step, c, t), places=15, msg=f"step {step}")
            self.assertEqual(trainer.lr_scheduler_configs[0].interval, "step")
            self.assertEqual(trainer.lr_scheduler_configs[0].scheduler.last_epoch, n_steps)

    def test_resume_in_cosine_phase_continues_on_curve(self):
        c, t = 2, 10
        with tempfile.TemporaryDirectory() as tmp:
            trainer, ckpt, rec = self._trainer(tmp, max_epochs=1)
            trainer.fit(self._module(lambda opt: ConstantCosineLR(opt, c, t, MIN_LR)))
            self.assertEqual(trainer.global_step, 4)  # inside the cosine phase (2 <= 4 < 10)

            # The resumed run builds a fresh optimizer/scheduler from config (lr back at LR), like
            # audio_train_resume.py, then Lightning restores their states from the checkpoint.
            trainer2, _, rec2 = self._trainer(tmp, max_epochs=4)
            trainer2.fit(self._module(lambda opt: ConstantCosineLR(opt, c, t, MIN_LR)), ckpt_path=ckpt.last_model_path)
            steps = [s for s, _ in rec2.records]
            self.assertEqual(steps, list(range(4, 16)))
            for step, lr in rec2.records:
                self.assertAlmostEqual(lr, self._expected(step, c, t), places=15, msg=f"step {step}")
            self.assertLess(rec2.records[0][1], LR, "resume must not restart the constant phase")

    def test_reduce_on_plateau_and_early_stopping_unchanged(self):
        from pytorch_lightning.callbacks import EarlyStopping

        with tempfile.TemporaryDirectory() as tmp:
            stop = EarlyStopping(monitor="val_loss", mode="min", patience=1)
            trainer, _, rec = self._trainer(tmp, max_epochs=2, callbacks=[stop])
            trainer.fit(self._module(lambda opt: ReduceLROnPlateau(opt, patience=10, factor=0.5)))
            cfg = trainer.lr_scheduler_configs[0]
            self.assertTrue(cfg.reduce_on_plateau)
            self.assertEqual((cfg.interval, cfg.monitor), ("epoch", "val_loss"))
            self.assertTrue(all(lr == LR for _, lr in rec.records))

    def test_resume_takes_plateau_and_early_stop_settings_from_config(self):
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping

        from look2hear.utils.checkpoint_utils import ApplyConfigOnResume

        class Seen(pl.Callback):
            def on_train_epoch_start(self, trainer, pl_module):
                self.patience = trainer.lr_scheduler_configs[0].scheduler.patience
                self.bad = trainer.lr_scheduler_configs[0].scheduler.num_bad_epochs
                self.es = [cb.patience for cb in trainer.callbacks if isinstance(cb, EarlyStopping)][0]

        with tempfile.TemporaryDirectory() as tmp:
            trainer, ckpt, _ = self._trainer(tmp, max_epochs=1, callbacks=[EarlyStopping("val_loss", patience=15)])
            trainer.fit(self._module(lambda opt: ReduceLROnPlateau(opt, patience=10, factor=0.5)))
            resume_from = os.path.join(tmp, "epoch0.ckpt")  # last.ckpt is overwritten by each resumed run
            shutil.copy(ckpt.last_model_path, resume_from)
            saved = torch.load(resume_from, map_location="cpu", weights_only=False)["lr_schedulers"][0]

            for apply_config, expected in ((False, (10, 15)), (True, (3, 5))):
                with self.subTest(apply_config=apply_config):
                    module = self._module(lambda opt: ReduceLROnPlateau(opt, patience=3, factor=0.5))
                    stop, seen = EarlyStopping("val_loss", patience=5), Seen()
                    extra = [stop, seen] + ([ApplyConfigOnResume(module.scheduler, stop)] if apply_config else [])
                    trainer2, _, _ = self._trainer(tmp, max_epochs=2, callbacks=extra)
                    trainer2.fit(module, ckpt_path=resume_from)
                    self.assertEqual((seen.patience, seen.es), expected)
                    self.assertEqual(seen.bad, saved["num_bad_epochs"])  # progress still from the checkpoint


if __name__ == "__main__":
    unittest.main()

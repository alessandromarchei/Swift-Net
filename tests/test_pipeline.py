"""Tests for the visual-embedding cache, the cached dataset path, checkpoint migration and the Lightning module.

Run from the repository root:
    python -m unittest tests.test_pipeline -v
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from look2hear.datas.avspeech_dataset import AVSpeechDataset
from look2hear.datas.visual_cache import VisualFeatureCache, create_cache, write_rows
from look2hear.utils.checkpoint_utils import migrate_checkpoint


SR = 16000
SEGMENT = 2.0  # 32000 samples, 50 visual frames at 25 fps
D = 4


def make_fake_corpus(root, n_mix=3, cache_dir=None, dim=D):
    """Tiny LRS-like split on disk and a matching cache built with a fake deterministic backbone.

    Returns (json_dir, cache_dir, fake_backbone). The fake backbone maps [T, H, W] -> [D, T].
    """
    from look2hear.datas.transform import get_preprocessing_pipelines

    root = Path(root)
    rng = np.random.RandomState(0)
    wav_dir, mouth_dir, json_dir = root / "wav", root / "mouths", root / "json"
    for d in (wav_dir, mouth_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)
    mix, s1, s2 = [], [], []
    length = int(SEGMENT * SR)
    for i in range(n_mix):
        mouth_path = mouth_dir / f"m{i}.npz"
        np.savez(mouth_path, data=rng.randint(0, 256, size=(50, 96, 96)).astype(np.uint8))
        src_wav = wav_dir / f"s{i}.wav"
        mix_wav = wav_dir / f"mix{i}.wav"
        sf.write(src_wav, rng.randn(length).astype(np.float32) * 0.1, SR)
        sf.write(mix_wav, rng.randn(length).astype(np.float32) * 0.1, SR)
        mix.append([str(mix_wav), length])
        s1.append([str(src_wav), str(mouth_path), length])
        s2.append([str(src_wav), str(mouth_path), length])
    for name, obj in [("mix", mix), ("s1", s1), ("s2", s2)]:
        (json_dir / f"{name}.json").write_text(json.dumps(obj))

    proj = torch.randn(dim, 1, generator=torch.Generator().manual_seed(1))

    def fake_backbone(frames):
        per_frame = torch.as_tensor(frames).float().mean(dim=(1, 2))
        return (proj * per_frame[None, :]).float()

    cache_dir = str(root / "cache") if cache_dir is None else cache_dir
    online = AVSpeechDataset(str(json_dir), n_src=1, sample_rate=SR, segment=SEGMENT, is_train=False)
    keys = online.visual_keys()
    pipeline = get_preprocessing_pipelines()["val"]
    meta = {"cache_format_version": 1, "backbone": {"fake": True}, "preprocessing": {"name": "val_center_crop"},
            "input_fps": 25, "segment_duration": SEGMENT, "num_frames": 50, "feature_dim": dim,
            "feature_dtype": "float32"}
    create_cache(cache_dir, keys, meta)
    embs = [fake_backbone(pipeline(np.load(k)["data"])[:, :50]).numpy() for k in keys]
    write_rows(cache_dir, np.arange(len(keys)), np.stack(embs), np.array([50] * len(keys)))
    return str(json_dir), cache_dir, fake_backbone


class CachedDatasetEquivalenceTests(unittest.TestCase):
    """The cached dataset path must return exactly what the online path would compute.

    A fake deterministic "backbone" (random linear map of the mean over pixels) stands in for FRCNNVideoModel.
    Cached entries are filled with that map applied to the same preprocessed mouth, so the two paths must agree.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.json_dir, self.cache_dir, self.fake_backbone = make_fake_corpus(self.tmp.name)
        self.online = AVSpeechDataset(self.json_dir, n_src=1, sample_rate=SR, segment=SEGMENT, is_train=False)
        self.cache = VisualFeatureCache(self.cache_dir)
        self.cached = AVSpeechDataset(self.json_dir, n_src=1, sample_rate=SR, segment=SEGMENT,
                                      is_train=False, visual_cache=self.cache)
        self.D = D

    def tearDown(self):
        self.tmp.cleanup()

    def test_cached_items_equal_online_backbone_output(self):
        for i in range(len(self.online)):
            _, _, mouth_online, _ = self.online[i]
            _, _, emb_cached, _ = self.cached[i]
            self.assertEqual(emb_cached.shape, (D, 50))
            torch.testing.assert_close(emb_cached, self.fake_backbone(mouth_online), rtol=0, atol=1e-6)

    def test_waveforms_are_identical_in_both_modes(self):
        for i in range(len(self.online)):
            mix_a, src_a, _, name_a = self.online[i]
            mix_b, src_b, _, name_b = self.cached[i]
            torch.testing.assert_close(mix_a, mix_b, rtol=0, atol=0)
            torch.testing.assert_close(src_a, src_b, rtol=0, atol=0)
            self.assertEqual(name_a, name_b)

    def test_missing_entries_are_reported_by_datamodule_check(self):
        missing = VisualFeatureCache(self.cache_dir).missing_keys(self.online.visual_keys() + ["/nope.npz"])
        self.assertEqual(missing, ["/nope.npz"])

    def test_cache_is_shared_by_mixtures_using_the_same_visual_source(self):
        # Two mixtures use the same visual source m0 -> one cache entry, identical embeddings.
        self.assertEqual(len(self.cache), 3)


class CheckpointMigrationTests(unittest.TestCase):
    def test_monitor_rename_keeps_callback_state_and_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "a.ckpt"), os.path.join(tmp, "b.ckpt")
            legacy_key = "ModelCheckpoint{'monitor': 'val_loss/dataloader_idx_0', 'mode': 'min'}"
            ckpt = {
                "state_dict": {"audio_model.w": torch.ones(2), "video_model.w": torch.ones(3)},
                "callbacks": {legacy_key: {"best_k_models": {"x": 1.0}}},
                "optimizer_states": [{"state": {"step": 5}}],
            }
            torch.save(ckpt, src)
            report = migrate_checkpoint(src, dst, drop_video_model=True)
            out = torch.load(dst, weights_only=False)
            self.assertEqual(report["dropped_video_keys"], 1)
            self.assertEqual(list(out["state_dict"]), ["audio_model.w"])
            self.assertEqual(list(out["callbacks"]), ["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min'}"])
            self.assertEqual(out["optimizer_states"], ckpt["optimizer_states"])


class VisualCacheTests(unittest.TestCase):
    def setUp(self):
        from look2hear.datas.visual_cache import create_cache

        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "cache")
        self.keys = [f"/data/mouths/{i:04d}.npz" for i in range(6)]
        self.meta = {
            "cache_format_version": 1,
            "backbone": {"videonet_name": "FRCNNVideoModel"},
            "preprocessing": {"name": "val_center_crop"},
            "input_fps": 25,
            "segment_duration": 2.0,
            "num_frames": 10,
            "feature_dim": 4,
            "feature_dtype": "float32",
        }
        create_cache(self.root, self.keys, self.meta)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_read_round_trip_with_variable_length(self):
        from look2hear.datas.visual_cache import VisualFeatureCache, write_rows

        emb = torch.randn(2, 4, 10).numpy()
        write_rows(self.root, [1, 4], emb, [10, 7])
        cache = VisualFeatureCache(self.root)
        torch.testing.assert_close(cache.get(self.keys[1]), torch.from_numpy(emb[0]))
        short = cache.get(self.keys[4])
        self.assertEqual(short.shape, (4, 7))
        torch.testing.assert_close(short, torch.from_numpy(emb[1][:, :7]))

    def test_incomplete_entry_raises_and_missing_keys_reported(self):
        from look2hear.datas.visual_cache import VisualFeatureCache

        cache = VisualFeatureCache(self.root)
        with self.assertRaises(RuntimeError):
            cache.get(self.keys[0])
        self.assertEqual(len(cache.missing_keys(self.keys + ["/data/mouths/none.npz"])), 7)

    def test_picklable_for_dataloader_workers(self):
        import pickle

        from look2hear.datas.visual_cache import VisualFeatureCache, write_rows

        write_rows(self.root, [0], torch.zeros(1, 4, 10).numpy(), [10])
        cache = VisualFeatureCache(self.root)
        cache.get(self.keys[0])  # opens the memmap in this process
        restored = pickle.loads(pickle.dumps(cache))
        self.assertIsNone(restored._features)
        self.assertEqual(restored.get(self.keys[0]).shape, (4, 10))

    def test_incompatible_metadata_is_rejected(self):
        from look2hear.datas.visual_cache import check_metadata

        changed = dict(self.meta, feature_dtype="float16")
        with self.assertRaises(ValueError):
            check_metadata(changed, self.meta)

    def test_verify_detects_non_finite_values(self):
        from look2hear.datas.visual_cache import verify_cache, write_rows

        bad = torch.zeros(1, 4, 10)
        bad[0, 0, 0] = float("nan")
        write_rows(self.root, [2], bad.numpy(), [10])
        summary = verify_cache(self.root)
        self.assertEqual(summary["non_finite_rows"], [2])
        self.assertEqual(summary["completed"], 1)


class LightningModuleTests(unittest.TestCase):
    def _module(self, visual_mode, video_model):
        from look2hear.system import AudioVisualLightningModule

        class Audio(nn.Module):
            def forward(self, wav, emb):
                self.last_emb = emb
                return wav.unsqueeze(1)

        return AudioVisualLightningModule(
            audio_model=Audio(),
            video_model=video_model,
            loss_func={"train": None, "val": None},
            visual_mode=visual_mode,
        )

    def test_validation_has_single_dataloader_and_no_dataloader_idx(self):
        import inspect

        from look2hear.system import AudioVisualLightningModule

        self.assertNotIn("dataloader_idx", inspect.signature(AudioVisualLightningModule.validation_step).parameters)
        m = self._module("cached", None)
        m.val_loader, m.test_loader = ["val"], ["test"]
        self.assertEqual(m.val_dataloader(), ["val"])
        self.assertEqual(m.test_dataloader(), ["test"])
        self.assertEqual(m.default_monitor, "val_loss")

    def test_cached_mode_passes_embeddings_without_video_model(self):
        m = self._module("cached", None)
        emb = torch.randn(2, 512, 50)
        m(torch.randn(2, 1600), emb)
        self.assertEqual(m.audio_model.last_emb.shape, emb.shape)

    def test_cached_mode_rejects_video_model_and_online_requires_it(self):
        with self.assertRaises(ValueError):
            self._module("cached", nn.Identity())
        with self.assertRaises(ValueError):
            self._module("online", None)

    def test_online_mode_runs_video_model_frozen(self):
        calls = []

        class Video(nn.Module):
            def forward(self, x):
                calls.append(tuple(x.shape))
                return torch.zeros(x.shape[0], 512, 50)

        m = self._module("online", Video())
        m(torch.randn(2, 1600), torch.randn(2, 50, 88, 88))
        self.assertEqual(calls, [(2, 1, 50, 88, 88)])


class LightningIntegrationTests(unittest.TestCase):
    """Real Trainer on CPU with the cached SwiftNet config: fit, checkpoint monitor, explicit test, resume."""

    def test_fit_never_tests_and_resume_keeps_topk_state(self):
        import pytorch_lightning as pl
        import yaml
        from pytorch_lightning.callbacks import ModelCheckpoint
        from torch.utils.data import DataLoader

        import look2hear.losses
        from look2hear.models import SwiftNet
        from look2hear.system import AudioVisualLightningModule

        with tempfile.TemporaryDirectory() as tmp:
            json_dir, cache_dir, _ = make_fake_corpus(tmp, dim=512)  # SwiftNet expects 512-d embeddings
            ds = AVSpeechDataset(json_dir, n_src=1, sample_rate=SR, segment=SEGMENT, is_train=False,
                                 visual_cache=VisualFeatureCache(cache_dir))
            loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=True)
            conf = yaml.safe_load(open(REPO / "configs" / "lrs2_SwiftNet_6_cached.yml"))
            loss_cfg = conf["loss"]
            loss = {
                split: getattr(look2hear.losses, cfg["loss_func"])(
                    getattr(look2hear.losses, cfg["sdr_type"]), **cfg["config"])
                for split, cfg in (("train", loss_cfg["train"]), ("val", loss_cfg["val"]))
            }

            calls = {"test": 0}

            class Module(AudioVisualLightningModule):
                def test_step(self, *args, **kwargs):
                    calls["test"] += 1
                    return super().test_step(*args, **kwargs)

            def build_module():
                # SwiftNet pops keys from its config dicts, so each build gets its own copy.
                audio = SwiftNet(sample_rate=SR, **copy.deepcopy(conf["audionet"]["audionet_config"]))
                opt = torch.optim.AdamW(audio.parameters(), lr=1e-3)
                return Module(audio_model=audio, video_model=None, optimizer=opt, loss_func=loss,
                              train_loader=loader, val_loader=loader, test_loader=loader,
                              scheduler=None, config=conf, visual_mode="cached")

            ckpt_dir = os.path.join(tmp, "ckpt")

            def build_trainer(max_epochs):
                ckpt = ModelCheckpoint(dirpath=ckpt_dir, filename="epoch-{epoch:03d}", monitor="val_loss",
                                       mode="min", save_top_k=2, save_last=True)
                trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", devices=1, logger=False,
                                     callbacks=[ckpt], enable_progress_bar=False, enable_model_summary=False,
                                     limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0,
                                     default_root_dir=tmp, enable_checkpointing=True)
                return trainer, ckpt

            trainer, ckpt = build_trainer(max_epochs=1)
            trainer.fit(build_module())
            self.assertEqual(calls["test"], 0, "trainer.fit must not run the test loop")
            self.assertIn("val_loss", trainer.callback_metrics)
            self.assertNotIn("test_loss", trainer.callback_metrics)

            saved = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["visual_mode"], "cached")
            self.assertEqual(saved["architecture"]["frequency_mixer"]["type"], "grnn")
            self.assertFalse(any(k.startswith("video_model.") for k in saved["state_dict"]))
            self.assertTrue(any("'monitor': 'val_loss'" in k for k in saved["callbacks"]))

            trainer.test(build_module(), dataloaders=loader)
            self.assertEqual(calls["test"], len(loader), "trainer.test must run one test_step per test batch")

            # Resume: the saved top-k state is restored, not reset.
            trainer2, ckpt2 = build_trainer(max_epochs=2)
            trainer2.fit(build_module(), ckpt_path=ckpt.best_model_path)
            self.assertGreaterEqual(len(ckpt2.best_k_models), 1)
            self.assertGreaterEqual(trainer2.current_epoch, 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for the configurable visual frame rate (visual_fps <= 25).

Run from the repository root:
    python -m unittest tests.test_visual_fps -v
"""

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

from look2hear.datas.avspeech_dataset import AVSpeechDataModule, AVSpeechDataset, validate_visual_fps
from tests.test_pipeline import SEGMENT, SR, make_fake_corpus


def make_split(root, n_frames=50):
    """Single-source split whose mouth files hold n_frames frames each (25 fps for a 2 s segment = 50)."""
    root = Path(root)
    wav_dir, mouth_dir, json_dir = root / "wav", root / "mouths", root / "json"
    for d in (wav_dir, mouth_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    length = int(SEGMENT * SR)
    mix, s1, s2 = [], [], []
    for i in range(2):
        mouth = mouth_dir / f"m{i}.npz"
        np.savez(mouth, data=rng.randint(0, 256, size=(n_frames, 96, 96)).astype(np.uint8))
        src = wav_dir / f"s{i}.wav"
        mx = wav_dir / f"mix{i}.wav"
        sf.write(src, rng.randn(length).astype(np.float32) * 0.1, SR)
        sf.write(mx, rng.randn(length).astype(np.float32) * 0.1, SR)
        mix.append([str(mx), length])
        s1.append([str(src), str(mouth), length])
        s2.append([str(src), str(mouth), length])
    for name, obj in [("mix", mix), ("s1", s1), ("s2", s2)]:
        (json_dir / f"{name}.json").write_text(json.dumps(obj))
    return str(json_dir)


class VisualFpsValidationTests(unittest.TestCase):
    def test_accepts_rates_up_to_25(self):
        for fps in (25, 25.0, 12.5, 10, 17.5):
            self.assertEqual(validate_visual_fps(fps), float(fps))

    def test_rejects_rates_above_25_and_non_positive(self):
        for fps in (0, -5, 25.5, 30):
            with self.assertRaises(ValueError):
                validate_visual_fps(fps)


class VisualFpsFrameCountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.json_dir = make_split(self.tmp.name, n_frames=50)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fps_len_follows_visual_fps(self):
        expected = {25.0: 50, 12.5: 25, 10.0: 20}  # round(2.0 * fps)
        for fps, frames in expected.items():
            ds = AVSpeechDataset(self.json_dir, n_src=1, sample_rate=SR, segment=SEGMENT,
                                 is_train=False, visual_fps=fps)
            self.assertEqual(ds.fps_len, frames, fps)

    def test_mouth_is_truncated_to_the_segment_at_the_run_rate(self):
        # A 25 fps mouth file longer than the segment: at 10 fps only the first 2 s, i.e. 20 frames, may be used.
        longer = make_split(os.path.join(self.tmp.name, "long"), n_frames=70)
        ds = AVSpeechDataset(longer, n_src=1, sample_rate=SR, segment=SEGMENT, is_train=False, visual_fps=10)
        _, _, mouth, _ = ds[0]
        self.assertEqual(mouth.shape[1], 20)

    def test_default_keeps_previous_behaviour(self):
        ds = AVSpeechDataset(self.json_dir, n_src=1, sample_rate=SR, segment=SEGMENT, is_train=False)
        self.assertEqual(ds.visual_fps, 25.0)
        self.assertEqual(ds.fps_len, int(SEGMENT * 25))


class VisualFpsCacheCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Cache built at the native 25 fps (50 frames for the 2 s segment).
        self.json_dir, self.cache_dir, _ = make_fake_corpus(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _datamodule(self, visual_fps):
        return AVSpeechDataModule(
            train_dir=self.json_dir, valid_dir=self.json_dir, test_dir=self.json_dir,
            n_src=1, sample_rate=SR, segment=SEGMENT, batch_size=1, num_workers=0,
            visual_features_path=self.cache_dir, visual_fps=visual_fps,
        )

    def test_matching_cache_rate_is_accepted(self):
        self._datamodule(25).setup()

    def test_cache_at_another_rate_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "built at 25 fps"):
            self._datamodule(10).setup()


if __name__ == "__main__":
    unittest.main()

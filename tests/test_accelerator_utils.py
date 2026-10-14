"""Tests for the accelerator option validation used by the training entry points.

Run from the repository root:
    python -m unittest tests.test_accelerator_utils -v

torch_xla is not required: the XLA branch is exercised with a stub module.
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from look2hear.utils.accelerator_utils import validate_accelerator_options


class TestValidateAcceleratorOptions(unittest.TestCase):
    def test_cuda_accepts_any_precision_and_compile(self):
        for precision in ("32-true", "bf16-mixed", "16-mixed"):
            validate_accelerator_options("cuda", precision, torch_compile_enabled=True)

    def test_unknown_accelerator_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_accelerator_options("mps", "32-true", False)

    def test_xla_without_torch_xla_fails_fast(self):
        with mock.patch.dict(sys.modules, {"torch_xla": None}):
            with self.assertRaises(RuntimeError):
                validate_accelerator_options("xla", "32-true", False)

    def test_xla_rejects_16_mixed(self):
        with mock.patch.dict(sys.modules, {"torch_xla": types.ModuleType("torch_xla")}):
            with self.assertRaises(ValueError):
                validate_accelerator_options("xla", "16-mixed", False)

    def test_xla_rejects_torch_compile(self):
        with mock.patch.dict(sys.modules, {"torch_xla": types.ModuleType("torch_xla")}):
            with self.assertRaises(ValueError):
                validate_accelerator_options("xla", "bf16-mixed", True)

    def test_xla_accepts_bf16_and_sets_tpu_runtime(self):
        env = {k: v for k, v in os.environ.items() if k != "PJRT_DEVICE"}
        with mock.patch.dict(sys.modules, {"torch_xla": types.ModuleType("torch_xla")}), \
                mock.patch.dict(os.environ, env, clear=True):
            validate_accelerator_options("xla", "bf16-mixed", False)
            self.assertEqual(os.environ["PJRT_DEVICE"], "TPU")

    def test_xla_rejects_true_half_precisions(self):
        with mock.patch.dict(sys.modules, {"torch_xla": types.ModuleType("torch_xla")}):
            for precision in ("bf16-true", "16-true"):
                with self.assertRaises(ValueError):
                    validate_accelerator_options("xla", precision, False)

    def test_xla_trainer_precision(self):
        try:
            from look2hear.utils.accelerator_utils import XLABf16MixedPrecision, xla_trainer_precision
            import torch_xla  # noqa: F401  (XLAPrecision needs it)
        except ImportError:
            self.skipTest("needs Lightning and torch_xla")
        self.assertEqual(xla_trainer_precision("32-true"), {"precision": "32-true"})
        plugins = xla_trainer_precision("bf16-mixed")["plugins"]
        self.assertIsInstance(plugins[0], XLABf16MixedPrecision)
        with plugins[0].forward_context():
            self.assertTrue(torch.is_autocast_enabled("xla"))
            self.assertEqual(torch.get_autocast_dtype("xla"), torch.bfloat16)

    def test_xla_keeps_existing_pjrt_device(self):
        with mock.patch.dict(sys.modules, {"torch_xla": types.ModuleType("torch_xla")}), \
                mock.patch.dict(os.environ, {"PJRT_DEVICE": "CPU"}):
            validate_accelerator_options("xla", "32-true", False)
            self.assertEqual(os.environ["PJRT_DEVICE"], "CPU")


if __name__ == "__main__":
    unittest.main()

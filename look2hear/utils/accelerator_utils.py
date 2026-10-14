"""Accelerator selection shared by the training entry points.

``cuda`` is the default GPU path (Lightning ``gpu`` accelerator with DDP).
``xla`` targets TPUs through torch_xla, for example a Kaggle TPU v5e-8: Lightning
starts one process per TPU core and selects the XLA strategy automatically.
"""

import os
import warnings

import contextlib

import torch

try:
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.plugins.precision.xla import XLAPrecision
except ImportError:  # only the Lightning classes below need it
    Callback = XLAPrecision = object

ACCELERATORS = ("cuda", "xla")


def set_xla_matmul_precision(tf32_enabled):
    """Precision of float32 matmuls and convolutions on TPU.

    torch_xla defaults to 'default': one bfloat16 pass, far less precise than float32 on GPU.
    'highest' (six passes, ~float32) matches a float32 GPU run; with --tf32 'high' (three
    passes, ~14 bits) is used, the TPU counterpart of TF32. Returns the precision set.
    """
    precision = "high" if tf32_enabled else "highest"
    try:
        import torch_xla.backends

        torch_xla.backends.set_mat_mul_precision(precision)
        return precision
    except (ImportError, AttributeError):
        pass

    import torch_xla

    xlac = torch_xla._XLAC
    if hasattr(xlac, "_xla_set_mat_mul_precision"):
        xlac._xla_set_mat_mul_precision(precision)
        return precision
    if hasattr(xlac, "_xla_set_use_full_mat_mul_precision"):
        # Older torch_xla: only full (float32) or default precision.
        xlac._xla_set_use_full_mat_mul_precision(use_full_mat_mul_precision=True)
        return "highest"
    warnings.warn("This torch_xla cannot set the matmul precision; TPU matmuls stay at bfloat16 precision.")
    return "default"


class XLABf16MixedPrecision(XLAPrecision):
    """'bf16-mixed' on XLA, which Lightning's XLAPrecision does not offer (only *-true modes).

    Same as Lightning's 'bf16-mixed' on GPU: float32 weights, gradients and optimizer, steps run
    under ``torch.autocast(..., dtype=torch.bfloat16, cache_enabled=False)``.
    """

    def __init__(self):
        super().__init__("32-true")
        self.precision = "bf16-mixed"

    @contextlib.contextmanager
    def forward_context(self):
        with torch.autocast("xla", dtype=torch.bfloat16, cache_enabled=False):
            yield


def xla_trainer_precision(precision):
    """Trainer kwargs for ``precision`` on XLA: Lightning's own flag, or the bf16-mixed plugin."""
    if precision == "bf16-mixed":
        return {"plugins": [XLABf16MixedPrecision()]}
    return {"precision": precision}


class XLAMatmulPrecision(Callback):
    """Sets the TPU matmul precision in every process: XLA runs one process per TPU core."""

    def __init__(self, tf32_enabled):
        super().__init__()
        self.tf32_enabled = tf32_enabled

    def setup(self, trainer, pl_module, stage):
        precision = set_xla_matmul_precision(self.tf32_enabled)
        if trainer.is_global_zero:
            print(f"XLA float32 matmul/conv precision: {precision}")


def validate_accelerator_options(accelerator, precision, torch_compile_enabled):
    """Fail fast on option combinations that do not work on the selected accelerator."""
    if accelerator not in ACCELERATORS:
        raise ValueError(
            f"Unknown accelerator '{accelerator}'. Choose from {ACCELERATORS}."
        )

    if accelerator != "xla":
        return

    try:
        import torch_xla  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "--accelerator xla requires torch_xla to be installed."
        ) from error

    if precision not in ("32-true", "bf16-mixed"):
        raise ValueError(
            f"'{precision}' is not supported on TPU. "
            "Use 'bf16-mixed' (native on TPU) or '32-true'."
        )

    if torch_compile_enabled:
        raise ValueError(
            "--torch_compile is CUDA-only: XLA already compiles the graph lazily."
        )

    # Lets torch_xla pick the TPU runtime without extra notebook setup.
    os.environ.setdefault("PJRT_DEVICE", "TPU")

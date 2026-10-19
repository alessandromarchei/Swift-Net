"""Accelerator selection shared by the training entry points.

``cuda`` is the default GPU path (Lightning ``gpu`` accelerator with DDP).
``xla`` targets TPUs through torch_xla, for example a Kaggle TPU v5e-8: Lightning
starts one process per TPU core and selects the XLA strategy automatically.
"""

import os

ACCELERATORS = ("cuda", "xla")


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

    if precision == "16-mixed":
        raise ValueError(
            "'16-mixed' is not supported on TPU. "
            "Use 'bf16-mixed' (native on TPU) or '32-true'."
        )

    if torch_compile_enabled:
        raise ValueError(
            "--torch_compile is CUDA-only: XLA already compiles the graph lazily."
        )

    # Lets torch_xla pick the TPU runtime without extra notebook setup.
    os.environ.setdefault("PJRT_DEVICE", "TPU")

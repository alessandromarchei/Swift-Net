#!/usr/bin/env python3
"""Export a finished Swift-Net Lightning checkpoint without resuming training.

The script reconstructs only the audio and video networks, loads their weights
from a Lightning ``.ckpt`` file, saves the serialized audio model expected by
``BaseModel.from_pretrain()``, and exports audio, video and end-to-end ONNX
graphs.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import onnx
import torch
import torch.nn as nn
import yaml

import look2hear.models
import look2hear.videomodels

class EndToEndSwiftNet(nn.Module):
    """Raw mouth frames -> video embedding -> separated waveform."""

    def __init__(self, video_model: nn.Module, audio_model: nn.Module) -> None:
        super().__init__()
        self.video_model = video_model
        self.audio_model = audio_model

    def forward(
        self,
        audio_mixture: torch.Tensor,
        mouth_video: torch.Tensor,
    ) -> torch.Tensor:
        mouth_embedding = self.video_model(mouth_video)
        return self.audio_model(audio_mixture, mouth_embedding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a completed Swift-Net Lightning checkpoint."
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        required=True,
        help="Best Lightning checkpoint (.ckpt) to export.",
    )
    parser.add_argument(
        "--conf_dir",
        type=Path,
        default=Path("configs/lrs2_SwiftNet_6.yml"),
        help="YAML configuration used for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory. By default, if the checkpoint is inside a "
            "checkpoints/ directory, its parent experiment directory is used."
        ),
    )
    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=2.0,
        help="Length of the example audio passed to the ONNX exporter.",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=25.0,
        help="Mouth-video frame rate used to calculate the example frame count.",
    )
    parser.add_argument("--mouth_height", type=int, default=88)
    parser.add_argument("--mouth_width", type=int, default=88)
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset. Opset 17 is recommended because SwiftNet uses STFT.",
    )
    parser.add_argument(
        "--dynamic_axes",
        action="store_true",
        help="Export dynamic batch, audio-length and video-time axes.",
    )
    parser.add_argument(
        "--skip_audio_onnx",
        action="store_true",
        help="Do not export the audio network with precomputed visual embedding.",
    )
    parser.add_argument(
        "--skip_video_onnx",
        action="store_true",
        help="Do not export the standalone video encoder.",
    )
    parser.add_argument(
        "--skip_full_onnx",
        action="store_true",
        help="Do not export the end-to-end audiovisual network.",
    )
    return parser.parse_args()


def infer_output_dir(checkpoint_path: Path) -> Path:
    checkpoint_dir = checkpoint_path.parent
    if checkpoint_dir.name == "checkpoints":
        return checkpoint_dir.parent
    return checkpoint_dir


def normalize_lightning_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remove wrappers that may have been introduced by torch.compile."""
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        normalized[key] = value
    return normalized


def extract_submodule_state(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    prefix_with_dot = f"{prefix}."
    result = {
        key[len(prefix_with_dot) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix_with_dot)
    }
    if not result:
        preview = list(state_dict)[:10]
        raise RuntimeError(
            f"No parameters beginning with '{prefix_with_dot}' were found. "
            f"First checkpoint keys: {preview}"
        )
    return result


def load_models(
    config: dict,
    checkpoint_path: Path,
) -> tuple[nn.Module, nn.Module]:
    sample_rate = config["datamodule"]["data_config"]["sample_rate"]

    audio_model = getattr(
        look2hear.models,
        config["audionet"]["audionet_name"],
    )(
        sample_rate=sample_rate,
        **copy.deepcopy(config["audionet"]["audionet_config"]),
    )

    # The checkpoint contains the complete video state. Avoid loading the old
    # pretrained backbone path, which may have moved since training.
    video_config = copy.deepcopy(config["videonet"]["videonet_config"])
    if "pretrain" in video_config:
        video_config["pretrain"] = None

    video_model = getattr(
        look2hear.videomodels,
        config["videonet"]["videonet_name"],
    )(**video_config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "state_dict" not in checkpoint:
        raise RuntimeError("The checkpoint does not contain a 'state_dict'.")

    state_dict = normalize_lightning_state_dict(checkpoint["state_dict"])
    audio_state = extract_submodule_state(state_dict, "audio_model")
    video_state = extract_submodule_state(state_dict, "video_model")

    audio_model.load_state_dict(audio_state, strict=True)
    video_model.load_state_dict(video_state, strict=True)
    audio_model.eval()
    video_model.eval()
    return audio_model, video_model


def export_onnx(
    model: nn.Module,
    example_inputs,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    dynamic_axes: dict | None,
) -> None:
    print(f"Exporting {output_path.name} ...")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example_inputs,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    exported_model = onnx.load(str(output_path))
    onnx.checker.check_model(exported_model)
    print(f"Validated: {output_path}")


def main() -> None:
    args = parse_args()
    checkpoint_path = args.ckpt_path.expanduser().resolve()
    config_path = args.conf_dir.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    if args.segment_seconds <= 0:
        raise ValueError("--segment_seconds must be positive.")
    if args.video_fps <= 0:
        raise ValueError("--video_fps must be positive.")

    with config_path.open("r") as file:
        config = yaml.safe_load(file)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else infer_output_dir(checkpoint_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Configuration: {config_path}")
    print(f"Output directory: {output_dir}")

    audio_model, video_model = load_models(config, checkpoint_path)
    print("Checkpoint loaded strictly into audio_model and video_model.")

    best_model_path = output_dir / "best_model.pth"
    torch.save(audio_model.serialize(), best_model_path)
    print(f"Serialized audio model saved: {best_model_path}")

    sample_rate = int(config["datamodule"]["data_config"]["sample_rate"])
    audio_samples = int(round(sample_rate * args.segment_seconds))
    video_frames = int(round(args.video_fps * args.segment_seconds))

    audio_mixture = torch.randn(1, audio_samples, dtype=torch.float32)
    mouth_video = torch.randn(
        1,
        1,
        video_frames,
        args.mouth_height,
        args.mouth_width,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        mouth_embedding = video_model(mouth_video)

    print(f"Example audio shape: {tuple(audio_mixture.shape)}")
    print(f"Example mouth-video shape: {tuple(mouth_video.shape)}")
    print(f"Derived mouth-embedding shape: {tuple(mouth_embedding.shape)}")

    audio_dynamic_axes = None
    video_dynamic_axes = None
    full_dynamic_axes = None
    if args.dynamic_axes:
        audio_dynamic_axes = {
            "audio_mixture": {0: "batch", 1: "audio_samples"},
            "mouth_embedding": {0: "batch", 2: "video_frames"},
            "separated_audio": {0: "batch", 2: "audio_samples"},
        }
        video_dynamic_axes = {
            "mouth_video": {0: "batch", 2: "video_frames"},
            "mouth_embedding": {0: "batch", 2: "video_frames"},
        }
        full_dynamic_axes = {
            "audio_mixture": {0: "batch", 1: "audio_samples"},
            "mouth_video": {0: "batch", 2: "video_frames"},
            "separated_audio": {0: "batch", 2: "audio_samples"},
        }

    if not args.skip_audio_onnx:
        export_onnx(
            model=audio_model,
            example_inputs=(audio_mixture, mouth_embedding),
            output_path=output_dir / "audio_model.onnx",
            input_names=["audio_mixture", "mouth_embedding"],
            output_names=["separated_audio"],
            opset=args.opset,
            dynamic_axes=audio_dynamic_axes,
        )

    if not args.skip_video_onnx:
        export_onnx(
            model=video_model,
            example_inputs=mouth_video,
            output_path=output_dir / "video_model.onnx",
            input_names=["mouth_video"],
            output_names=["mouth_embedding"],
            opset=args.opset,
            dynamic_axes=video_dynamic_axes,
        )

    if not args.skip_full_onnx:
        full_model = EndToEndSwiftNet(video_model, audio_model).eval()
        export_onnx(
            model=full_model,
            example_inputs=(audio_mixture, mouth_video),
            output_path=output_dir / "full_model.onnx",
            input_names=["audio_mixture", "mouth_video"],
            output_names=["separated_audio"],
            opset=args.opset,
            dynamic_axes=full_dynamic_axes,
        )

    print("Export completed successfully.")


if __name__ == "__main__":
    main()
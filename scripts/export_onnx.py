#!/usr/bin/env python3
"""Export SwiftNet graphs for inspection in Netron.

The optimized ``sru`` package uses custom C++/CUDA operators that cannot be
exported to ONNX.  This script replaces each optimized SRU with an opaque ONNX
node named ``swiftnet::SRU``.  STFT and iSTFT are deliberately kept outside the
ONNX graph because the legacy exporter cannot handle PyTorch complex tensors.

Generated files:
  * video_model.onnx       : raw mouth frames -> visual embedding
  * audio_core.onnx        : real/imag/magnitude STFT + visual embedding -> mask output
  * complete_av_core.onnx  : real/imag/magnitude STFT + raw video -> mask output

These files are intended for architectural inspection, not numerical inference:
the opaque SRU replacements preserve input/output sizes but not trained SRU
weights or numerical behavior.
"""

import argparse
import os
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sru import SRU as OptimizedSRU

# Suppress import-time deprecation warnings before importing the repository.
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import look2hear.models
import look2hear.videomodels


# Keep exporter output readable. Actual exceptions are still printed.
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.onnx").setLevel(logging.ERROR)


def create_high_level_overview(
    output_path,
    audio_model,
    dummy_audio,
    dummy_video,
    stft_components,
    audio_embedding,
    audio_features,
    mouth_embedding,
    mask_output,
    decoder_output,
    opset_version=17,
):
    """
    Crea un ONNX schematico per Netron.

    Ogni nodo rappresenta un intero blocco SwiftNet.
    Non contiene le operazioni interne e non è destinato
    all'inferenza numerica.
    """
    import onnx
    from onnx import TensorProto, helper

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator = audio_model.separator
    n_src = int(audio_model.n_src)

    def tensor_shape(tensor):
        return [
            int(dimension)
            for dimension in tensor.shape
        ]

    audio_feature_shape = tensor_shape(
        audio_features
    )

    video_feature_shape = tensor_shape(
        mouth_embedding
    )

    nodes = []
    intermediate_values = []

    def add_block(
        block_type,
        inputs,
        outputs,
        output_shapes,
        **attributes,
    ):
        """
        Aggiunge un nodo custom:

            domain = swiftnet
            op_type = nome del blocco
        """

        nodes.append(
            helper.make_node(
                block_type,
                inputs=inputs,
                outputs=outputs,
                name=block_type,
                domain="swiftnet",
                **attributes,
            )
        )

        for output_name, output_shape in zip(
            outputs,
            output_shapes,
        ):
            intermediate_values.append(
                helper.make_tensor_value_info(
                    output_name,
                    TensorProto.FLOAT,
                    output_shape,
                )
            )

    # ========================================================
    # Audio frontend
    # ========================================================

    add_block(
        block_type="STFT",
        inputs=[
            "audio_waveform",
        ],
        outputs=[
            "stft_components",
        ],
        output_shapes=[
            tensor_shape(stft_components),
        ],
        n_fft_i=int(audio_model.encoder.win),
        hop_length_i=int(
            audio_model.encoder.hop_length
        ),
        representation_s=(
            "real_imaginary_magnitude"
        ),
    )

    add_block(
        block_type="AudioEncoder",
        inputs=[
            "stft_components",
        ],
        outputs=[
            "audio_embedding",
        ],
        output_shapes=[
            tensor_shape(audio_embedding),
        ],
        operation_s="causal_Conv2D",
    )

    add_block(
        block_type="AudioBottleneck",
        inputs=[
            "audio_embedding",
        ],
        outputs=[
            "audio_features_0",
        ],
        output_shapes=[
            audio_feature_shape,
        ],
        operation_s="cLNhw_ReLU_Conv1x1",
    )

    # ========================================================
    # Video frontend
    # ========================================================

    add_block(
        block_type="VisionEncoder",
        inputs=[
            "video_frames",
        ],
        outputs=[
            "video_features_0",
        ],
        output_shapes=[
            video_feature_shape,
        ],
        architecture_s="Conv3D_ResNet18",
        output_channels_i=int(
            mouth_embedding.shape[1]
        ),
    )

    audio_current = "audio_features_0"
    video_current = "video_features_0"

    # Questi sono i residual originali del Separator.
    audio_residual = "audio_features_0"
    video_residual = "video_features_0"

    # ========================================================
    # Iterazioni multimodali:
    #
    # FTGS || LightVid
    #        ↓
    #       SAF
    # ========================================================

    for index in range(
        separator.fusion_repeats
    ):
        repeat_number = index + 1

        # Nel codice originale il residual viene aggiunto
        # dalla seconda iterazione in avanti.
        if index > 0:
            audio_with_residual = (
                f"audio_residual_input_"
                f"{repeat_number}"
            )

            video_with_residual = (
                f"video_residual_input_"
                f"{repeat_number}"
            )

            add_block(
                block_type=(
                    f"AudioResidualAdd_"
                    f"{repeat_number}"
                ),
                inputs=[
                    audio_current,
                    audio_residual,
                ],
                outputs=[
                    audio_with_residual,
                ],
                output_shapes=[
                    audio_feature_shape,
                ],
            )

            add_block(
                block_type=(
                    f"VideoResidualAdd_"
                    f"{repeat_number}"
                ),
                inputs=[
                    video_current,
                    video_residual,
                ],
                outputs=[
                    video_with_residual,
                ],
                output_shapes=[
                    video_feature_shape,
                ],
            )

            audio_current = audio_with_residual
            video_current = video_with_residual

        ftgs_output = (
            f"audio_after_FTGS_{repeat_number}"
        )

        lightvid_output = (
            f"video_after_LightVid_"
            f"{repeat_number}"
        )

        add_block(
            block_type=f"FTGS_{repeat_number}",
            inputs=[
                audio_current,
            ],
            outputs=[
                ftgs_output,
            ],
            output_shapes=[
                audio_feature_shape,
            ],
            shared_weights_i=int(
                separator.audio_shared
            ),
            causal_i=int(audio_model.causal),
        )

        add_block(
            block_type=(
                f"LightVid_{repeat_number}"
            ),
            inputs=[
                video_current,
            ],
            outputs=[
                lightvid_output,
            ],
            output_shapes=[
                video_feature_shape,
            ],
            shared_weights_i=int(
                separator.video_shared
            ),
            hidden_channels_i=64,
        )

        fused_audio = (
            f"audio_after_SAF_{repeat_number}"
        )

        fused_video = (
            f"video_after_SAF_{repeat_number}"
        )

        # Nel MultiModalFusion originale tutti i blocchi
        # tranne l'ultimo possono fondere anche audio->video.
        video_fusion = (
            index
            != separator.fusion_repeats - 1
        )

        add_block(
            block_type=(
                f"SAF_Fusion_{repeat_number}"
            ),
            inputs=[
                ftgs_output,
                lightvid_output,
            ],
            outputs=[
                fused_audio,
                fused_video,
            ],
            output_shapes=[
                audio_feature_shape,
                video_feature_shape,
            ],
            video_to_audio_i=1,
            audio_to_video_i=int(video_fusion),
        )

        audio_current = fused_audio
        video_current = fused_video

    # ========================================================
    # FTGS audio-only dopo le fusioni AV
    # ========================================================

    for local_index in range(
        separator.audio_repeats
    ):
        absolute_number = (
            separator.fusion_repeats
            + local_index
            + 1
        )

        # Nel codice originale:
        #
        # i = j + fusion_repeats
        # audio_net(audio + audio_residual if i > 0 ...)
        #
        if absolute_number > 1:
            audio_with_residual = (
                f"audio_residual_input_"
                f"{absolute_number}"
            )

            add_block(
                block_type=(
                    f"AudioResidualAdd_"
                    f"{absolute_number}"
                ),
                inputs=[
                    audio_current,
                    audio_residual,
                ],
                outputs=[
                    audio_with_residual,
                ],
                output_shapes=[
                    audio_feature_shape,
                ],
            )

            audio_current = audio_with_residual

        ftgs_output = (
            f"audio_after_FTGS_"
            f"{absolute_number}"
        )

        add_block(
            block_type=(
                f"FTGS_{absolute_number}"
            ),
            inputs=[
                audio_current,
            ],
            outputs=[
                ftgs_output,
            ],
            output_shapes=[
                audio_feature_shape,
            ],
            audio_only_i=1,
            shared_weights_i=int(
                separator.audio_shared
            ),
            causal_i=int(audio_model.causal),
        )

        audio_current = ftgs_output

    # ========================================================
    # Ricostruzione
    # ========================================================

    add_block(
        block_type="MaskGenerator",
        inputs=[
            audio_current,

            # Skip connection dall'encoder audio originale.
            "audio_embedding",
        ],
        outputs=[
            "separated_spectral_embedding",
        ],
        output_shapes=[
            tensor_shape(mask_output),
        ],
        number_of_sources_i=n_src,
    )

    add_block(
        block_type="SpectralDecoder",
        inputs=[
            "separated_spectral_embedding",
        ],
        outputs=[
            "real_imaginary_spectrum",
        ],
        output_shapes=[
            tensor_shape(decoder_output),
        ],
        operation_s=(
            "causal_ConvTranspose2D"
        ),
    )

    separated_audio_shape = [
        int(dummy_audio.shape[0]),
        n_src,
        int(dummy_audio.shape[-1]),
    ]

    # Per l'ultimo output non serve registrare anche
    # un value_info intermedio.
    nodes.append(
        helper.make_node(
            "iSTFT",
            inputs=[
                "real_imaginary_spectrum",
            ],
            outputs=[
                "separated_audio",
            ],
            name="iSTFT",
            domain="swiftnet",
            n_fft_i=int(audio_model.decoder.win),
            hop_length_i=int(
                audio_model.decoder.hop_length
            ),
        )
    )

    # ========================================================
    # ONNX graph
    # ========================================================

    graph = helper.make_graph(
        nodes=nodes,
        name="SwiftNetHighLevel",

        inputs=[
            helper.make_tensor_value_info(
                "audio_waveform",
                TensorProto.FLOAT,
                tensor_shape(dummy_audio),
            ),
            helper.make_tensor_value_info(
                "video_frames",
                TensorProto.FLOAT,
                tensor_shape(dummy_video),
            ),
        ],

        outputs=[
            helper.make_tensor_value_info(
                "separated_audio",
                TensorProto.FLOAT,
                separated_audio_shape,
            ),
        ],

        value_info=intermediate_values,
    )

    onnx_model = helper.make_model(
        graph,
        producer_name=(
            "SwiftNet high-level exporter"
        ),
        opset_imports=[
            helper.make_operatorsetid(
                "",
                opset_version,
            ),
            helper.make_operatorsetid(
                "swiftnet",
                1,
            ),
        ],
    )

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(output_path))

    print()
    print(
        "High-level overview saved:"
    )
    print(f"  {output_path.resolve()}")

    print(
        f"  fusion repeats: "
        f"{separator.fusion_repeats}"
    )

    print(
        f"  audio-only FTGS repeats: "
        f"{separator.audio_repeats}"
    )


class OpaqueSRUFunction(torch.autograd.Function):
    """Execute a shape-compatible projection and export one opaque SRU node."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: [sequence, batch, input_size]
        y = F.linear(x, weight, bias)
        # GRNN ignores the returned state, but the pip SRU API returns it.
        h = y[-1:].contiguous()
        return y, h

    @staticmethod
    def symbolic(g, x, weight, bias):
        y, h = g.op(
            "swiftnet::SRU",
            x,
            weight,
            bias,
            outputs=2,
        )
        return y, h


class OpaqueSRU(nn.Module):
    """Shape-compatible stand-in for the optimized pip ``sru.SRU`` module."""

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.output_size = self.hidden_size * (2 if self.bidirectional else 1)

        # These dummy parameters make the eager forward shape-correct. In ONNX
        # they become inputs of the single opaque swiftnet::SRU node.
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.output_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, c0=None):
        del c0
        return OpaqueSRUFunction.apply(x, self.weight, self.bias)


def replace_all_srus(module, prefix=""):
    """Recursively replace every optimized SRU and return its full module name."""
    replaced = []

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, OptimizedSRU):
            replacement = OpaqueSRU(
                input_size=child.input_size,
                hidden_size=child.hidden_size,
                num_layers=child.num_layers,
                bidirectional=child.bidirectional,
            )
            setattr(module, name, replacement)
            replaced.append(full_name)
            print(
                f"  {full_name}: input={child.input_size}, "
                f"hidden={child.hidden_size}, layers={child.num_layers}, "
                f"bidirectional={child.bidirectional}"
            )
        else:
            replaced.extend(replace_all_srus(child, full_name))

    return replaced


def exportable_multiscale_pool(x, output_size):
    """ONNX-friendly replacement for adaptive average pooling.

    SwiftNet uses adaptive pooling only to bring every pyramid level to the
    deepest level. Its downsampling layers use stride 2 with same-like padding,
    so repeated average pooling with ``ceil_mode=True`` reproduces the required
    ceil-division shapes, including odd input dimensions. A final nearest
    resize is retained only as a defensive fallback for unusual YAML configs.
    """
    target = tuple(int(value) for value in output_size)
    y = x

    if x.dim() == 4:
        while y.shape[-2] > target[-2] or y.shape[-1] > target[-1]:
            y = F.avg_pool2d(
                y,
                kernel_size=2,
                stride=2,
                ceil_mode=True,
                count_include_pad=False,
            )
    elif x.dim() == 3:
        while y.shape[-1] > target[-1]:
            y = F.avg_pool1d(
                y,
                kernel_size=2,
                stride=2,
                ceil_mode=True,
                count_include_pad=False,
            )
    else:
        raise ValueError(f"Unsupported pooling input rank: {x.dim()}")

    if tuple(y.shape[-len(target) :]) != target:
        y = F.interpolate(y, size=target, mode="nearest")

    return y


def patch_adaptive_pools(module, prefix=""):
    """Replace FTGS adaptive pool callables only in the export model."""
    patched = []

    current_name = prefix or module.__class__.__name__
    if module.__class__.__name__ == "FTGS" and hasattr(module, "pool"):
        module.pool = exportable_multiscale_pool
        patched.append(current_name)

    for child_name, child in module.named_children():
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        patched.extend(patch_adaptive_pools(child, full_name))

    return patched


class VideoWrapper(nn.Module):
    def __init__(self, video_model):
        super().__init__()
        self.video_model = video_model

    def forward(self, video_frames):
        return self.video_model(video_frames)


class AudioCoreWrapper(nn.Module):
    """SwiftNet neural core without waveform STFT and iSTFT."""

    def __init__(self, audio_model):
        super().__init__()
        self.encoder_conv = audio_model.encoder.conv
        self.audio_bottleneck = audio_model.audio_bottleneck
        self.video_bottleneck = audio_model.video_bottleneck
        self.separator = audio_model.separator
        self.mask_generator = audio_model.mask_generator

    def forward(self, stft_components, mouth_embedding):
        # stft_components: [B, 3, T_audio, F]
        # channels: real, imaginary, magnitude
        audio_embedding = self.encoder_conv(stft_components)
        audio_features = self.audio_bottleneck(audio_embedding)
        video_features = self.video_bottleneck(mouth_embedding)
        refined_features = self.separator(audio_features, video_features)
        return self.mask_generator(refined_features, audio_embedding)


class CompleteAVCoreWrapper(nn.Module):
    """Video frontend plus the complete neural audio core."""

    def __init__(self, audio_model, video_model):
        super().__init__()
        self.video_model = video_model
        self.audio_core = AudioCoreWrapper(audio_model)

    def forward(self, stft_components, video_frames):
        mouth_embedding = self.video_model(video_frames)
        return self.audio_core(stft_components, mouth_embedding)


class SpectralDecoderWrapper(nn.Module):
    """Decoder convolution only: separated embedding -> real/imag spectrum."""

    def __init__(self, decoder):
        super().__init__()
        self.in_chan = decoder.in_chan
        self.n_src = decoder.n_src
        self.kernel_size = decoder.kernel_size
        self.padding = decoder.padding
        self.causal = decoder.causal
        self.decoder_conv = decoder.decoder

    def forward(self, separated_embedding):
        batch_size = separated_embedding.shape[0]
        x = separated_embedding.reshape(
            batch_size * self.n_src,
            self.in_chan,
            separated_embedding.shape[-2],
            separated_embedding.shape[-1],
        )
        if self.causal:
            x = F.pad(x, (0, 0, self.kernel_size - 1, 0))
        x = self.decoder_conv(x)
        if self.causal:
            x = x[:, :, : -self.padding, :]
        return x


def first_repeated_block(value):
    """Return a shared block or the first element of a ModuleList."""
    return value[0] if isinstance(value, nn.ModuleList) else value


def compute_ftgs_internal_tensors(ftgs, audio_features):
    """Run the FTGS pyramid up to its three global processing sub-blocks."""
    residual = ftgs.gateway(audio_features)
    encoded = ftgs.projection(residual)

    pyramid = [ftgs.downsample_layers[0](encoded)]
    for index in range(1, ftgs.upsampling_depth):
        pyramid.append(ftgs.downsample_layers[index](pyramid[-1]))

    target_shape = pyramid[-1].shape[-2:]
    global_features = sum(ftgs.pool(level, output_size=target_shape) for level in pyramid)
    frequency_output = ftgs.process_f(global_features)
    temporal_output = ftgs.process_t(frequency_output)
    attention_output = ftgs.process_att(temporal_output)

    return global_features, frequency_output, temporal_output, attention_output


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def instantiate_models(config):
    sample_rate = int(config["datamodule"]["data_config"]["sample_rate"])

    audio_section = config["audionet"]
    video_section = config["videonet"]

    audio_class = getattr(look2hear.models, audio_section["audionet_name"])
    video_class = getattr(look2hear.videomodels, video_section["videonet_name"])

    audio_model = audio_class(
        sample_rate=sample_rate,
        **audio_section["audionet_config"],
    )
    video_model = video_class(**video_section["videonet_config"])

    # Do not write ``video_model = video_model.eval()``: the repository's
    # FRCNNVideoModel.train() override does not return self.
    audio_model.cpu()
    audio_model.eval()
    video_model.cpu()
    video_model.eval()

    return audio_model, video_model, sample_rate


def make_stft_components(audio_model, waveform):
    """Run only preprocessing outside ONNX and return [real, imag, magnitude]."""
    encoder = audio_model.encoder
    waveform = encoder.unsqueeze_to_2D(waveform)

    spectrum = torch.stft(
        waveform,
        n_fft=encoder.win,
        hop_length=encoder.hop_length,
        window=encoder.window.to(waveform.device),
        return_complex=True,
    )
    magnitude = (spectrum.abs().pow(2) + encoder.eps).sqrt()

    return torch.stack(
        [spectrum.real, spectrum.imag, magnitude], dim=1
    ).transpose(2, 3).contiguous()


def validate_onnx(path):
    try:
        import onnx

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        print("  validation: OK")
    except ImportError:
        print("  validation: skipped (install package 'onnx')")
    except Exception as error:
        raise RuntimeError(f"ONNX validation failed for {path}: {error}") from error


@contextmanager
def capture_native_output(log_path):
    """Capture Python plus C/C++ stdout/stderr emitted by the ONNX exporter."""
    log_path = Path(log_path)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)

    try:
        with open(log_path, "w", encoding="utf-8") as log_stream:
            os.dup2(log_stream.fileno(), 1)
            os.dup2(log_stream.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def export_legacy(model, inputs, path, input_names, output_names, opset):
    """Export with TorchScript so the custom autograd symbolic is honored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    print(f"\nExporting {path.name} ...", flush=True)
    exporter_log = Path(f"{path}.export.log")

    try:
        with capture_native_output(exporter_log):
            with torch.inference_mode():
                torch.onnx.export(
                    model,
                    inputs,
                    str(path),
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamo=False,
                    verbose=False,
                )
    except Exception:
        print(f"  FAILED. Full exporter log: {exporter_log.resolve()}")
        raise
    else:
        exporter_log.unlink(missing_ok=True)

    print(f"  saved: {path.resolve()}")
    validate_onnx(path)


def print_shape(name, tensor):
    print(f"  {name:<28} {tuple(tensor.shape)}")


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    config = load_yaml(args.conf_dir)
    audio_model, video_model, sample_rate = instantiate_models(config)

    print("Replacing optimized SRUs with opaque ONNX nodes:")
    replaced = replace_all_srus(audio_model)
    if not replaced:
        raise RuntimeError("No pip sru.SRU modules were found in SwiftNet")
    print(f"Replaced SRUs: {len(replaced)}")

    print("Replacing FTGS adaptive pooling with exportable average pooling:")
    patched_pools = patch_adaptive_pools(audio_model)
    if not patched_pools:
        raise RuntimeError("No FTGS adaptive pooling callable was found")
    for module_name in patched_pools:
        print(f"  {module_name}")
    print(f"Patched FTGS pools: {len(patched_pools)}")

    audio_samples = (
        args.audio_samples
        if args.audio_samples is not None
        else int(round(args.audio_seconds * sample_rate))
    )
    if audio_samples < config["audionet"]["audionet_config"]["enc_dec_params"]["win"]:
        raise ValueError("Dummy audio must be at least one STFT window long")

    dummy_audio = torch.randn(args.batch_size, audio_samples)
    dummy_video = torch.randn(
        args.batch_size,
        1,
        args.video_frames,
        args.video_height,
        args.video_width,
    )

    with torch.inference_mode():
        stft_components = make_stft_components(audio_model, dummy_audio)
        mouth_embedding = video_model(dummy_video)

    print("\nDummy tensor shapes:")
    print_shape("waveform (outside ONNX)", dummy_audio)
    print_shape("STFT [real,imag,mag]", stft_components)
    print_shape("video frames", dummy_video)
    print_shape("mouth embedding", mouth_embedding)

    video_wrapper = VideoWrapper(video_model)
    audio_wrapper = AudioCoreWrapper(audio_model)
    complete_wrapper = CompleteAVCoreWrapper(audio_model, video_model)
    ftgs = first_repeated_block(audio_model.separator.audio_net)
    lightvid = first_repeated_block(audio_model.separator.video_net)
    fusion_block = audio_model.separator.caf.get_fusion_block(0)
    decoder_wrapper = SpectralDecoderWrapper(audio_model.decoder)

    video_wrapper.eval()
    audio_wrapper.eval()
    complete_wrapper.eval()
    ftgs.eval()
    lightvid.eval()
    fusion_block.eval()
    decoder_wrapper.eval()

    # Produce real intermediate shapes from the instantiated model. These are
    # then reused as dummy inputs for every standalone block export.
    with torch.inference_mode():
        video_output = video_wrapper(dummy_video)
        audio_embedding = audio_model.encoder.conv(stft_components)
        audio_features = audio_model.audio_bottleneck(audio_embedding)
        ftgs_output = ftgs(audio_features)
        lightvid_output = lightvid(mouth_embedding)
        fusion_audio_output, fusion_video_output = fusion_block(
            ftgs_output, lightvid_output
        )
        mask_output = audio_model.mask_generator(
            fusion_audio_output, audio_embedding
        )
        decoder_output = decoder_wrapper(mask_output)
        (
            ftgs_global_input,
            ftgs_frequency_output,
            ftgs_temporal_output,
            ftgs_attention_output,
        ) = compute_ftgs_internal_tensors(ftgs, audio_features)
        audio_output = audio_wrapper(stft_components, mouth_embedding)
        complete_output = complete_wrapper(stft_components, dummy_video)

    print("\nImportant intermediate shapes:")
    print_shape("video output", video_output)
    print_shape("audio encoder output", audio_embedding)
    print_shape("audio bottleneck output", audio_features)
    print_shape("FTGS output", ftgs_output)
    print_shape("FTGS global input", ftgs_global_input)
    print_shape("frequency block output", ftgs_frequency_output)
    print_shape("temporal block output", ftgs_temporal_output)
    print_shape("attention block output", ftgs_attention_output)
    print_shape("LightVid output", lightvid_output)
    print_shape("SAF audio output", fusion_audio_output)
    print_shape("SAF video output", fusion_video_output)
    print_shape("mask generator output", mask_output)
    print_shape("decoder real/imag output", decoder_output)
    print_shape("audio core output", audio_output)
    print_shape("complete core output", complete_output)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_path = (
        output_dir
        / "00_swiftnet_high_level.onnx"
    )

    create_high_level_overview(
        output_path=overview_path,
        audio_model=audio_model,
        dummy_audio=dummy_audio,
        dummy_video=dummy_video,
        stft_components=stft_components,
        audio_embedding=audio_embedding,
        audio_features=audio_features,
        mouth_embedding=mouth_embedding,
        mask_output=mask_output,
        decoder_output=decoder_output,
        opset_version=args.opset,
    )


    generated_files = []

    if not args.blocks_only:
        full_exports = [
            (
                video_wrapper,
                (dummy_video,),
                "video_model.onnx",
                ["video_frames"],
                ["mouth_embedding"],
            ),
            (
                audio_wrapper,
                (stft_components, mouth_embedding),
                "audio_core.onnx",
                ["stft_real_imag_magnitude", "mouth_embedding"],
                ["separated_spectral_embedding"],
            ),
            (
                complete_wrapper,
                (stft_components, dummy_video),
                "complete_av_core.onnx",
                ["stft_real_imag_magnitude", "video_frames"],
                ["separated_spectral_embedding"],
            ),
        ]
        for model, inputs, filename, input_names, output_names in full_exports:
            export_legacy(
                model,
                inputs,
                output_dir / filename,
                input_names,
                output_names,
                args.opset,
            )
            generated_files.append(output_dir / filename)

    blocks_dir = output_dir / "blocks"
    block_exports = [
        (
            video_wrapper,
            (dummy_video,),
            "01_vision_encoder.onnx",
            ["video_frames"],
            ["mouth_embedding"],
        ),
        (
            audio_model.encoder.conv,
            (stft_components,),
            "02_audio_encoder_conv.onnx",
            ["stft_real_imag_magnitude"],
            ["audio_embedding"],
        ),
        (
            audio_model.audio_bottleneck,
            (audio_embedding,),
            "03_audio_bottleneck.onnx",
            ["audio_embedding"],
            ["audio_features"],
        ),
        (
            ftgs,
            (audio_features,),
            "04_ftgs_complete.onnx",
            ["audio_features"],
            ["refined_audio_features"],
        ),
        (
            ftgs.process_f,
            (ftgs_global_input,),
            "04a_ftgs_frequency_modeling.onnx",
            ["multiscale_audio_features"],
            ["frequency_refined_features"],
        ),
        (
            ftgs.process_t,
            (ftgs_frequency_output,),
            "04b_ftgs_temporal_modeling.onnx",
            ["frequency_refined_features"],
            ["temporal_refined_features"],
        ),
        (
            ftgs.process_att,
            (ftgs_temporal_output,),
            "04c_ftgs_causal_attention.onnx",
            ["temporal_refined_features"],
            ["attention_refined_features"],
        ),
        (
            lightvid,
            (mouth_embedding,),
            "05_lightvid.onnx",
            ["mouth_embedding"],
            ["refined_video_features"],
        ),
        (
            fusion_block,
            (ftgs_output, lightvid_output),
            "06_saf_fusion.onnx",
            ["audio_features", "video_features"],
            ["fused_audio_features", "fused_video_features"],
        ),
        (
            audio_model.mask_generator,
            (fusion_audio_output, audio_embedding),
            "07_mask_generator.onnx",
            ["refined_audio_features", "mixture_audio_embedding"],
            ["separated_spectral_embedding"],
        ),
        (
            decoder_wrapper,
            (mask_output,),
            "08_spectral_decoder_conv.onnx",
            ["separated_spectral_embedding"],
            ["real_imaginary_spectrum"],
        ),
    ]

    for model, inputs, filename, input_names, output_names in block_exports:
        export_legacy(
            model,
            inputs,
            blocks_dir / filename,
            input_names,
            output_names,
            args.opset,
        )
        generated_files.append(blocks_dir / filename)

    print("\nDone. Open these files in Netron:")
    for path in generated_files:
        print(f"  {path.resolve()}")
    print("\nNote: STFT/iSTFT are external and every SRU is one opaque swiftnet::SRU node.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SwiftNet ONNX graphs for architectural inspection in Netron"
    )
    parser.add_argument(
        "--conf-dir",
        default="configs/lrs2_SwiftNet_6.yml",
        help="SwiftNet YAML configuration",
    )
    parser.add_argument(
        "--output-dir",
        default="Experiments/onnx_dummy",
        help="Output directory",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-seconds", type=float, default=2.0)
    parser.add_argument(
        "--audio-samples",
        type=int,
        default=None,
        help="Exact sample count; overrides --audio-seconds",
    )
    parser.add_argument("--video-frames", type=int, default=50)
    parser.add_argument("--video-height", type=int, default=96)
    parser.add_argument("--video-width", type=int, default=96)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--blocks-only",
        action="store_true",
        help="Export only the small block-level ONNX files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the complete traceback instead of a one-line error",
    )
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help=(
            "Generate only the small high-level "
            "SwiftNet overview"
        ),
    )
        
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        main(parsed_args)
    except Exception as error:
        if parsed_args.debug:
            raise

        first_line = str(error).splitlines()[0] if str(error) else "No error message"
        print(f"\nERROR [{type(error).__name__}]: {first_line}", file=sys.stderr)
        print("Use --debug for the complete traceback.", file=sys.stderr)
        raise SystemExit(1)
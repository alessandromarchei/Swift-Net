import argparse
from collections import OrderedDict

import torch


def extract_state_dict(checkpoint):
    """
    Cerca automaticamente lo state_dict nei formati più comuni.
    """

    if isinstance(checkpoint, OrderedDict):
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint non riconosciuto: {type(checkpoint)}"
        )

    possible_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "backbone",
    ]

    for key in possible_keys:
        if key in checkpoint and isinstance(
            checkpoint[key], (dict, OrderedDict)
        ):
            print(f"State dict trovato nella chiave: '{key}'")
            return checkpoint[key]

    # Potrebbe essere direttamente un dizionario nome → tensore
    if all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        print("Il checkpoint è direttamente uno state_dict.")
        return checkpoint

    raise KeyError(
        "Non trovo uno state_dict nel checkpoint.\n"
        f"Chiavi disponibili: {list(checkpoint.keys())}"
    )


def classify_weight(name, tensor):
    if not name.endswith("weight"):
        return None

    if tensor.ndim == 5:
        return "Conv3d"

    if tensor.ndim == 4:
        return "Conv2d"

    if tensor.ndim == 3:
        return "Conv1d"

    if tensor.ndim == 2:
        return "Linear"

    if tensor.ndim == 1:
        return "Norm/Bias/Scale"

    return "Altro"


def describe_weight(name, tensor):
    layer_type = classify_weight(name, tensor)

    print(f"{name}")
    print(f"    shape: {tuple(tensor.shape)}")
    print(f"    dtype: {tensor.dtype}")

    if layer_type is not None:
        print(f"    tipo probabile: {layer_type}")

    if layer_type == "Conv3d":
        out_channels, in_per_group, kt, kh, kw = tensor.shape

        print(f"    output channels: {out_channels}")
        print(f"    input channels per group: {in_per_group}")
        print(f"    temporal kernel: {kt}")
        print(f"    spatial kernel: {kh} x {kw}")

    elif layer_type == "Conv2d":
        out_channels, in_per_group, kh, kw = tensor.shape

        print(f"    output channels: {out_channels}")
        print(f"    input channels per group: {in_per_group}")
        print(f"    spatial kernel: {kh} x {kw}")

    elif layer_type == "Conv1d":
        out_channels, in_per_group, kernel = tensor.shape

        print(f"    output channels: {out_channels}")
        print(f"    input channels per group: {in_per_group}")
        print(f"    kernel: {kernel}")

    elif layer_type == "Linear":
        out_features, in_features = tensor.shape

        print(f"    output features: {out_features}")
        print(f"    input features: {in_features}")


def main(checkpoint_path):
    print(f"Caricamento checkpoint:\n{checkpoint_path}\n")

    # weights_only=False serve per checkpoint PyTorch meno recenti
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    print("=" * 100)
    print("STRUTTURA ESTERNA DEL CHECKPOINT")
    print("=" * 100)

    print(f"Tipo: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print("Chiavi principali:")

        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                description = (
                    f"Tensor {tuple(value.shape)}"
                )
            else:
                description = type(value).__name__

            print(f"  {key}: {description}")

    state_dict = extract_state_dict(checkpoint)

    print()
    print("=" * 100)
    print("TUTTI I PARAMETRI")
    print("=" * 100)

    total_parameters = 0

    for index, (name, tensor) in enumerate(
        state_dict.items()
    ):
        if not isinstance(tensor, torch.Tensor):
            print(
                f"{index:4d} | {name:70s} | "
                f"{type(tensor).__name__}"
            )
            continue

        number = tensor.numel()
        total_parameters += number

        print(
            f"{index:4d} | "
            f"{name:70s} | "
            f"{str(tuple(tensor.shape)):25s} | "
            f"{number:12,d}"
        )

    print()
    print(f"Numero tensori: {len(state_dict):,}")
    print(f"Numero totale parametri: {total_parameters:,}")

    print()
    print("=" * 100)
    print("PESI CONVOLUTIONAL E LINEAR")
    print("=" * 100)

    interesting_weights = []

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        layer_type = classify_weight(name, tensor)

        if layer_type in {
            "Conv3d",
            "Conv2d",
            "Conv1d",
            "Linear",
        }:
            interesting_weights.append(
                (name, tensor)
            )

            describe_weight(name, tensor)
            print()

    print()
    print("=" * 100)
    print("DEDUZIONE DEL POSSIBILE INPUT")
    print("=" * 100)

    convolution_weights = [
        (name, tensor)
        for name, tensor in interesting_weights
        if tensor.ndim in {3, 4, 5}
    ]

    if not convolution_weights:
        print("Nessun peso convolutional trovato.")
        return

    first_name, first_weight = convolution_weights[0]

    print(f"Prima convoluzione trovata: {first_name}")
    print(f"Forma: {tuple(first_weight.shape)}")
    print()

    if first_weight.ndim == 5:
        out_channels, input_channels, kt, kh, kw = (
            first_weight.shape
        )

        print("Il primo layer sembra una Conv3d.")
        print()
        print(
            "La Conv3d PyTorch si aspetta normalmente:"
        )
        print()
        print("    [B, C, T, H, W]")
        print()
        print(f"C dedotto dai pesi: {input_channels}")
        print(f"Kernel temporale: {kt}")
        print(f"Kernel spaziale: {kh} × {kw}")

        if input_channels == 1:
            print()
            print(
                "Il modello probabilmente usa frame grayscale."
            )
            print(
                "Possibile input: [B, 1, T, H, W]"
            )

    elif first_weight.ndim == 4:
        out_channels, input_channels, kh, kw = (
            first_weight.shape
        )

        print("Il primo layer sembra una Conv2d.")
        print()
        print(
            "La Conv2d PyTorch si aspetta normalmente:"
        )
        print()
        print("    [B, C, H, W]")
        print()
        print(f"C dedotto dai pesi: {input_channels}")
        print(f"Kernel spaziale: {kh} × {kw}")
        print()
        print(
            "Se il modello elabora una sequenza video, "
            "potrebbe ricevere [B,T,H,W], trasformarla "
            "internamente in [B*T,C,H,W] oppure aspettarsi "
            "direttamente [B*T,C,H,W]."
        )

    print()
    print(
        "Nota: H e W non sono necessariamente deducibili "
        "dai pesi. Il nome 'frcnn_128_512' suggerisce "
        "probabilmente crop 128×128 ed embedding da 512."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=(
            "/home/ale/work/av-tse/Swift-Net/"
            "pretrain_zoo/frcnn_128_512.backbone.pth.tar"
        ),
    )

    args = parser.parse_args()
    main(args.checkpoint)
import torch
import pickle
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from typing import Dict, Any, Tuple

from src.models.ml_coarsening_bce_module import MLCoarseningBCEModule
from src.models.binary_classification_module import BinaryClassificationModule
from src.models.ml_coarsening_module import MLCoarseningModule
from src.models.multi_classification_module import MulticlassClassificationModule


def load_model_from_checkpoint(ckpt_path: str, model_class: str):
    """Load model using the same logic as inference.py"""
    if model_class == "BinaryClassificationModule":
        return BinaryClassificationModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"),
                                                               strict=False)
    elif model_class == "MLCoarseningBCEModule":
        return MLCoarseningBCEModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
    elif model_class == "MulticlassClassificationModule":
        return MulticlassClassificationModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
    else:
        return MLCoarseningModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))


def export_model_parameters_as_text(model, output_path: Path, scaler=None):
    """
    Export model parameters in a flat text format as required for C++ implementation.
    First line: parameter count
    Following lines: individual parameter values (one per line)
    """
    # Get all parameters as a single flattened vector
    param_vector = torch.nn.utils.parameters_to_vector(model.parameters())

    # Write to file in the requested format
    param_file = output_path / 'model_parameters.txt'
    with open(param_file, "w") as f:
        f.write("{")
        for i, p in enumerate(param_vector):
            f.write(f"{p.item()}")
            if i < len(param_vector) - 1:
                f.write(",")
        f.write("\n")
        f.write("};")

    print(f"Exported {param_vector.size(0)} model parameters to {param_file}")

    # Detect network architecture in more detail
    architecture_info = []
    activation_map = {
        torch.nn.ReLU: "ReLU",
        torch.nn.LeakyReLU: "LeakyReLU",
        torch.nn.Sigmoid: "Sigmoid",
        torch.nn.Tanh: "Tanh",
        torch.nn.GELU: "GELU",
        torch.nn.ELU: "ELU",
        torch.nn.SiLU: "SiLU",
        torch.nn.Softmax: "Softmax",
        torch.nn.Dropout: "Dropout"
    }

    # Create a list of all modules in order to determine the sequence
    all_modules = list(model.modules())

    # Start with basic model class info
    architecture_info.append(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    architecture_info.append("\nLayer structure:")

    # Track sequential structure
    for i, module in enumerate(all_modules):
        if isinstance(module, torch.nn.Linear):
            in_feat = module.in_features
            out_feat = module.out_features
            bias = module.bias is not None
            param_count = in_feat * out_feat + (out_feat if bias else 0)
            architecture_info.append(f"Linear layer: in={in_feat}, out={out_feat}, bias={bias}, params={param_count}")

            # Look ahead for the activation function (if any)
            if i + 1 < len(all_modules):
                next_module = all_modules[i + 1]
                for act_class, act_name in activation_map.items():
                    if isinstance(next_module, act_class):
                        if isinstance(next_module, torch.nn.LeakyReLU):
                            architecture_info.append(
                                f"  Activation: {act_name} (negative_slope={next_module.negative_slope})")
                        elif isinstance(next_module, torch.nn.Dropout):
                            architecture_info.append(f"  Dropout: p={next_module.p}")
                        else:
                            architecture_info.append(f"  Activation: {act_name}")

    # Write a comprehensive C++ friendly description
    with open(output_path / 'network_architecture.txt', 'w') as f:
        f.write("MODEL ARCHITECTURE\n\n")
        f.write('\n'.join(architecture_info))

        # Add scaler information if available
        if scaler is not None:
            f.write("\n\nSCALER INFORMATION\n")
            f.write(f"Scaler type: {type(scaler).__name__}\n")

            # List available scaler parameters
            f.write("Available parameters:\n")
            if hasattr(scaler, 'mean_'):
                f.write(f"- mean_: shape={scaler.mean_.shape}\n")
            if hasattr(scaler, 'scale_'):
                f.write(f"- scale_: shape={scaler.scale_.shape}\n")
            if hasattr(scaler, 'var_'):
                f.write(f"- var_: shape={scaler.var_.shape}\n")
            if hasattr(scaler, 'min_'):
                f.write(f"- min_: shape={scaler.min_.shape}\n")
            if hasattr(scaler, 'max_'):
                f.write(f"- max_: shape={scaler.max_.shape}\n")

        f.write("\n\nParameter shape information:\n")

        # Add information about parameter shapes and ordering
        for name, param in model.named_parameters():
            f.write(f"{name}: shape={list(param.shape)}, size={param.numel()}\n")


def export_model_components(ckpt_path: str, model_class: str, output_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Export scaler and model weights from checkpoint"""
    # Load model
    model = load_model_from_checkpoint(ckpt_path, model_class)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scaler = model.hparams.scaler

    export_model_parameters_as_text(model, output_path, scaler)

    # Export model state dict
    model_weights = {}
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            model_weights[key] = value.cpu().numpy()

    if scaler is not None:
        export_scaler_params(scaler, output_path)

    # Export features
    features = model.hparams.features
    if features is not None:
        np.savetxt(output_path / 'features.txt', features, fmt='%s')

    hyperparams = dict(model.hparams)
    print(f"Exported components to {output_path}")
    print(f"Model weights shape: {len(model_weights)} parameters")
    print(f"Features: {len(features) if features is not None else 'None'}")
    print(f"Scaler type: {type(scaler).__name__ if scaler is not None else 'None'}")

    return model_weights, hyperparams


def export_scaler_params(scaler, output_path: Path):
    """Export scaler parameters as separate numpy files"""
    scaler_params = {}

    if hasattr(scaler, 'mean_'):
        scaler_params['means'] = scaler.mean_

    if hasattr(scaler, 'scale_'):
        scaler_params['stdevs'] = scaler.scale_
    elif hasattr(scaler, 'var_'):
        scaler_params['stdevs'] = np.sqrt(scaler.var_)

    # Save as text files for C++
    if scaler_params:
        for key, value in scaler_params.items():
            np.savetxt(output_path / f'scaler_{key}.txt', value)


@hydra.main(version_base="1.3", config_path="../configs", config_name="export_model.yaml")
def main(cfg: DictConfig):
    weights, hyperparams = export_model_components(
        ckpt_path=cfg.ckpt_path,
        model_class=cfg.model_class,
        output_dir=cfg.output_dir
    )


if __name__ == "__main__":
    main()
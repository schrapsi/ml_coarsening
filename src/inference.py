import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import hydra

from src.models.ml_coarsening_bce_module import MLCoarseningBCEModule
from src.models.binary_classification_module import BinaryClassificationModule
from src.models.ml_coarsening_module import MLCoarseningModule
from lightning import LightningDataModule
from omegaconf import DictConfig
from lightning.pytorch import Trainer

from src.models.multi_classification_module import MulticlassClassificationModule


def inference(cfg: DictConfig):
    model = load_model(cfg.ckpt_path, model_class=cfg.model_class)

    graph_file_path = Path(cfg.data.graphs_file)
    graph_set_name = graph_file_path.stem

    ckpt_path = Path(cfg.ckpt_path)
    print(f"Checkpoint path: {ckpt_path}")
    model_dir = ckpt_path.parent.parent
    print(f"Model directory: {model_dir}")
    pred_dir = Path(model_dir) / "predictions" / graph_set_name
    print(f"Prediction directory: {pred_dir}")
    pred_dir.mkdir(exist_ok=True, parents=True)

    model_name = model_dir.name[11:]

    print(f"Model name: {model_name}")
    create_experiment_json_file(
        name=model_name,
        instance_folder=str(pred_dir),
        output_path=str(model_dir),
        graph_set_name=graph_set_name
    )

    scaler = model.hparams.scaler
    features = model.hparams.features
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, scaler=scaler, features=features)

    datamodule.setup(stage="predict")
    print(f"Loaded model from {cfg.ckpt_path}")

    dataloaders = datamodule.predict_dataloader()
    trainer: Trainer = Trainer()
    copy_metis_files(cfg.metis_path, pred_dir, dataloaders.keys())
    for graph in dataloaders:
        dl = dataloaders[graph]
        print(f"Predict dataloader: {graph}, with {len(dl.dataset)} samples")
        raw_outputs = trainer.predict(model, dataloaders=dl)
        ids = raw_outputs[0]["ids"]  # shape [B, 2]
        preds = raw_outputs[0]["preds"]
        write_to_file(pred_dir, graph, ids, preds)


def create_experiment_json_file(name: str, instance_folder: str, output_path: str, graph_set_name: str):
    file_path = "/nfs/home/schrape/hypergraph_partitioner/experiments/experiment.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    data["name"] = name
    data["graph_instance_folder"] = instance_folder

    output_path = Path(output_path) / f"predictions/{graph_set_name}_experiment.json"
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

def write_to_file(path, graph_name, ids, preds):
    out_path = Path(path) / f"{graph_name}.metis.freq.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)

    # exactly your old CSV format:
    with open(out_path, "w") as f:
        f.write("id_high_degree,id_low_degree,frequency\n")
        f.write("# max=1\n")
        for (hi, lo), p in zip(ids, preds):
            f.write(f"{int(hi)},{int(lo)},{float(p)}\n")

    print(f"Wrote {out_path}")


def copy_metis_files(src_folder, dest_folder, graph_set):
    # Iterate through files in the source folder
    for graph in graph_set:
        filename = graph + ".metis"
        src_path = os.path.join(src_folder, filename)
        dest_path = os.path.join(dest_folder, filename)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)  # copy2 preserves metadata
            print(f"Copied: {src_path} -> {dest_path}")
        else:
            print(f"File not found: {src_path}, skipping copy.")


def load_model(ckpt_path, model_class=None):
    """Load model from checkpoint by detecting the model class from checkpoint."""
    if model_class == "BinaryClassificationModule" :
        print("Loading BinaryClassificationModule from checkpoint")
        return BinaryClassificationModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"), strict=False)
    elif model_class == "MLCoarseningBCEModule":
        print("Loading MLCoarseningBCEModule from checkpoint")
        return MLCoarseningBCEModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
    elif model_class == "MulticlassClassificationModule":
        print("Loading MulticlassClassificationModule from checkpoint")
        return MulticlassClassificationModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
    # Default or fallback to MLCoarseningModule
    print("Loading MLCoarseningModule from checkpoint")
    return MLCoarseningModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    inference(cfg)

    return None


if __name__ == "__main__":
    main()

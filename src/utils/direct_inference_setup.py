import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from omegaconf import DictConfig
import hydra

from src.inference import copy_metis_files


def main(cfg):
    graphs_file = Path(cfg.data.graphs_file)
    graph_set_name = graphs_file.stem

    model_dir = Path(cfg.model_dir)
    print(f"Model directory: {model_dir}")
    exp_sets_dir = Path(model_dir) / "experiment_sets" / graph_set_name
    print(f"Experiment sets directory: {exp_sets_dir}")
    exp_sets_dir.mkdir(exist_ok=True, parents=True)
    print(f"Created experiment sets directory at: {exp_sets_dir}")

    model_name = model_dir.name[11:]
    file_path = "/nfs/home/schrape/ml_inside.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    data["name"] = model_name
    data["graph_instance_folder"] = str(exp_sets_dir)

    output_path = Path(str(model_dir)) / f"experiment_sets/{graph_set_name}_experiment.json"
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

    graph_list = []
    if graphs_file and Path(graphs_file).exists():
        with open(graphs_file, 'r') as f:
            graph_list = [line.strip() for line in f if line.strip()]
    copy_metis_files(cfg.metis_path, exp_sets_dir, graph_list)


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print("starting main")
    main(cfg)

    return None

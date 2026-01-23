import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from omegaconf import DictConfig
import hydra

from src.inference import copy_metis_files


def setup(cfg):
    graphs_file = Path(cfg.data.graphs_file)
    graph_set_name = graphs_file.stem

    model_dir = Path(cfg.model_dir)
    output_dir = model_dir / "experiment_sets"
    output_dir.mkdir(exist_ok=True, parents=True)

    exp_sets_dir = output_dir / graph_set_name
    exp_sets_dir.mkdir(exist_ok=True, parents=True)

    model_name = model_dir.name[11:]
    file_path = "/nfs/home/schrape/ml_inside.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    for algo in data["config"]:
        algo["args"] = str(cfg.get("flags"))
    data["name"] = model_name
    data["graph_instance_folder"] = str(exp_sets_dir)

    output_path = output_dir / f"{graph_set_name}_experiment.json"
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

    graph_list = []
    if graphs_file and Path(graphs_file).exists():
        with open(graphs_file, 'r') as f:
            graph_list = [line.strip() for line in f if line.strip()]
    copy_metis_files(cfg.data.data_dir, exp_sets_dir, graph_list)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    setup(cfg)

    return None


if __name__ == "__main__":
    main()


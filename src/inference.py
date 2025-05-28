from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import hydra
from models.ml_coarsening_module import MLCoarseningModule
from lightning import LightningDataModule
from omegaconf import DictConfig
from lightning.pytorch import Trainer


def inference(cfg: DictConfig):
    model = MLCoarseningModule.load_from_checkpoint(cfg.ckpt_path)
    scaler = model.hparams.scaler
    features = model.hparams.features
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, scaler=scaler, features=features)

    datamodule.setup(stage="predict")
    print(f"Loaded model from {cfg.ckpt_path}")

    dataloaders = datamodule.predict_dataloader()
    trainer: Trainer = Trainer()
    for graph in dataloaders:
        dl = dataloaders[graph]
        print(f"Predict dataloader: {graph}, with {len(dl.dataset)} samples")
        raw_outputs = trainer.predict(model, dataloaders=dl)
        ids = raw_outputs[0]["ids"]  # shape [B, 2]
        preds = raw_outputs[0]["preds"]
        write_to_file(cfg.pred_path, graph, ids, preds)


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


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    inference(cfg)

    return None


if __name__ == "__main__":
    main()

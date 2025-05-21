from typing import Optional, Tuple, Dict, Any
import hydra
from omegaconf import DictConfig
from lightning import LightningDataModule, LightningModule, Trainer


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    #log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }
    if cfg.get("train"):
        #log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    #extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    #metric_value = get_metric_value(
    #    metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    #)

    # return optimized metric
    #return metric_value


if __name__ == "__main__":
    main()

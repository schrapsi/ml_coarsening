import os
from typing import Optional, Tuple, Dict, Any, List
import hydra
from omegaconf import DictConfig
from lightning import LightningDataModule, LightningModule, Trainer, Callback
from lightning.pytorch.loggers import Logger
from hydra.utils import to_absolute_path
import joblib


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        #log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks configs must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            #log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    #log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    print("Printing cfg.data:")
    print(cfg.data)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    cfg.model.net.input_size = datamodule.get_feature_count()

    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.hparams.scaler = datamodule.scaler

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger = None
    if not cfg.get("logger") is None:
        logger: Logger = hydra.utils.instantiate(cfg.logger.mlflow) or None

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }
    if cfg.get("train"):
        #log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    train_metrics = trainer.callback_metrics

    return {}, {}


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

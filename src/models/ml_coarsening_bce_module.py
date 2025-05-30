from typing import Any, Dict, Tuple

import hydra
import torch
from lightning import LightningModule
import torchmetrics as tm
from torchmetrics.regression import MeanSquaredError, R2Score, MeanAbsoluteError


class MLCoarseningBCEModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            scaler=None,
            features=None,
            compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        # BCE with Logits Loss
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Train metrics - using regression metrics for continuous targets
        self.train_loss = tm.MeanMetric()
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.train_r2 = R2Score()

        # Validation metrics
        self.val_loss = tm.MeanMetric()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()

        # Test metrics
        self.test_loss = tm.MeanMetric()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()

        # Best metric tracking
        self.val_mse_best = tm.MinMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Raw logits output from network
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        # Get probability between 0 and 1 by applying sigmoid
        return torch.sigmoid(self.forward(x))

    def on_train_start(self) -> None:
        # Reset metrics at start of training
        self.train_loss.reset()
        self.train_mse.reset()
        self.train_mae.reset()
        self.train_r2.reset()

        self.val_loss.reset()
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_r2.reset()

        self.val_mse_best.reset()

    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Convert logits to probabilities for evaluation metrics
        probs = torch.sigmoid(logits)
        return loss, probs, y

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, predictions, targets = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_mse(predictions, targets)
        self.train_mae(predictions, targets)
        self.train_r2(predictions, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, predictions, targets = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_mse(predictions, targets)
        self.val_mae(predictions, targets)
        self.val_r2(predictions, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        mse = self.val_mse.compute()  # Get current val MSE
        self.val_mse_best(mse)  # Update best so far (lowest) val MSE
        self.log("val/mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, predictions, targets = self.model_step(batch)

        # Update and log metrics
        self.test_loss(loss)
        self.test_mse(predictions, targets)
        self.test_mae(predictions, targets)
        self.test_r2(predictions, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=False)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def predict_step(self, batch, batch_idx: int):
        ids, features = batch
        logits = self(features)
        probs = torch.sigmoid(logits)  # Convert to probability between 0 and 1
        return {
            "ids": ids.cpu().numpy(),
            "preds": probs.squeeze(dim=-1).cpu().numpy()
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        if 'net' not in kwargs and 'net' in checkpoint['hyper_parameters']:
            kwargs['net'] = hydra.utils.instantiate(checkpoint['hyper_parameters']['net'])

        return super(MLCoarseningBCEModule, cls).load_from_checkpoint(
            checkpoint_path, map_location=map_location, **kwargs
        )
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
import torchmetrics as tm
from torchmetrics.regression import MeanSquaredError, R2Score


class MLCoarseningModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

        # Loss function
        self.criterion = torch.nn.MSELoss()

        # Metrics for regression
        # Train metrics
        self.train_mse = MeanSquaredError()
        self.train_r2 = R2Score()
        self.train_loss = tm.MeanMetric()

        # Validation metrics
        self.val_mse = MeanSquaredError()
        self.val_r2 = R2Score()
        self.val_loss = tm.MeanMetric()

        # Test metrics
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()
        self.test_loss = tm.MeanMetric()


        self.val_mse_best = tm.MinMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        # Reset metrics at start of training
        self.train_loss.reset()
        self.train_mse.reset()
        self.train_r2.reset()

        self.val_loss.reset()
        self.val_mse.reset()
        self.val_r2.reset()

        self.val_mse_best.reset()

    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, predictions, targets = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_mse(predictions, targets)
        self.train_r2(predictions, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, predictions, targets = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_mse(predictions, targets)
        self.val_r2(predictions, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        mse = self.val_mse.compute()  # Get current val MSE
        self.val_mse_best(mse)  # Update best so far (lowest) val MSE
        self.log("val/mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, predictions, targets = self.model_step(batch)

        # Update and log metrics
        self.test_loss(loss)
        self.test_mse(predictions, targets)
        self.test_r2(predictions, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True)

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

if __name__ == "__main__":
    _ = MLCoarseningModule(None, None, None, None)
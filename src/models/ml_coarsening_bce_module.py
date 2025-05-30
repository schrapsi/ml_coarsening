from typing import Any, Dict, Tuple

import hydra
import torch
from lightning import LightningModule
import torchmetrics as tm
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
)


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

        # Train metrics
        self.train_loss = tm.MeanMetric()
        self.train_acc = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.train_auroc = BinaryAUROC()

        # Validation metrics
        self.val_loss = tm.MeanMetric()
        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_auroc = BinaryAUROC()

        # Test metrics
        self.test_loss = tm.MeanMetric()
        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_auroc = BinaryAUROC()

        # Best metric tracking
        self.val_f1_best = tm.MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Raw logits output from network
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        # Get probability between 0 and 1 by applying sigmoid
        return torch.sigmoid(self.forward(x))

    def on_train_start(self) -> None:
        # Reset metrics at start of training
        self.train_loss.reset()
        self.train_acc.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
        self.train_auroc.reset()

        self.val_loss.reset()
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()

        self.val_f1_best.reset()

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
        self.train_acc(predictions, targets)
        self.train_precision(predictions, targets)
        self.train_recall(predictions, targets)
        self.train_f1(predictions, targets)
        self.train_auroc(predictions, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, predictions, targets = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_acc(predictions, targets)
        self.val_precision(predictions, targets)
        self.val_recall(predictions, targets)
        self.val_f1(predictions, targets)
        self.val_auroc(predictions, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        f1 = self.val_f1.compute()  # Get current val F1
        self.val_f1_best(f1)  # Update best so far (highest) val F1
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, predictions, targets = self.model_step(batch)

        # Update and log metrics
        self.test_loss(loss)
        self.test_acc(predictions, targets)
        self.test_precision(predictions, targets)
        self.test_recall(predictions, targets)
        self.test_f1(predictions, targets)
        self.test_auroc(predictions, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True)

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
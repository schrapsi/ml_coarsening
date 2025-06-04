from typing import Any, Dict, Tuple

import hydra
import torch
from lightning import LightningModule
import torchmetrics as tm
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall


class BinaryClassificationModule(LightningModule):
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

        # Binary classification loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Train metrics
        self.train_acc = Accuracy(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.train_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.train_loss = tm.MeanMetric()

        # Validation metrics
        self.val_acc = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_loss = tm.MeanMetric()

        # Test metrics
        self.test_acc = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_loss = tm.MeanMetric()

        # Keep track of best F1 score
        self.val_f1_best = tm.MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        # Reset metrics at start of training
        self.train_loss.reset()
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()

        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

        self.val_f1_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        # Make sure logits output is compatible with BCEWithLogitsLoss
        logits = logits.view(-1)
        y = y.float()  # Convert to float for BCEWithLogitsLoss
        loss = self.criterion(logits, y)

        preds = torch.sigmoid(logits)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
                      ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        f1 = self.val_f1.compute()  # Get current val F1
        self.val_f1_best(f1)  # Update best so far (highest) val F1
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)

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
        """
        Returns predictions as binary class (0 or 1)
        """
        ids, features = batch
        logits = self(features)  # -> Tensor [B, 1]
        preds = torch.sigmoid(logits.squeeze()) >= 0.5  # Apply threshold
        return {
            "ids": ids.cpu().numpy(),
            "preds": preds.int().cpu().numpy()  # Binary predictions as 0 or 1
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        """Override to automatically instantiate the network from saved config."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # If net is not explicitly provided but exists in hyperparameters
        if 'net' not in kwargs and 'net' in checkpoint['hyper_parameters']:
            # Create network from saved config
            kwargs['net'] = hydra.utils.instantiate(checkpoint['hyper_parameters']['net'])

        # Call the parent implementation with the network included
        return super(BinaryClassificationModule, cls).load_from_checkpoint(
            checkpoint_path, map_location=map_location, **kwargs
        )
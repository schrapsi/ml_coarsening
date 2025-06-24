from typing import Any, Dict, Tuple

import hydra
import torch
from lightning import LightningModule
from kornia.losses import FocalLoss
import torchmetrics as tm
from torchmetrics.classification import (
    Accuracy, F1Score, Precision, Recall, ConfusionMatrix
)
import matplotlib.pyplot as plt
import seaborn as sns


class MulticlassClassificationModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            num_classes: int = 10,
            scaler=None,
            features=None,
            compile: bool = False,
            class_weights=None,
            focal_gamma: float = 2.0
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.num_classes = num_classes

        # Multiclass classification loss function
        if class_weights is not None:
            self.criterion = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='focal_loss',
            alpha=class_weights,
            gamma=2,
            reduction='mean',
            dtype=torch.float32,
            force_reload=False)
        else:
            self.criterion = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='focal_loss',
                alpha=1.0,
                gamma=2,
                reduction='mean',
                dtype=torch.float32,
                force_reload=False)

        # Train metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.train_loss = tm.MeanMetric()

        # Validation metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_loss = tm.MeanMetric()
        self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # Test metrics
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_loss = tm.MeanMetric()
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

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
        self.val_confmat.reset()

        self.val_f1_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)  # [batch_size, num_classes]
        loss = self.criterion(logits, y)

        # Get predicted class probabilities (softmax)
        probs = torch.softmax(logits, dim=1)
        return loss, probs, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
                      ) -> torch.Tensor:
        loss, probs, targets = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_acc(probs, targets)
        self.train_f1(probs, targets)
        self.train_precision(probs, targets)
        self.train_recall(probs, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, probs, targets = self.model_step(batch)

        # Update and log metrics
        self.val_loss(loss)
        self.val_acc(probs, targets)
        self.val_f1(probs, targets)
        self.val_precision(probs, targets)
        self.val_recall(probs, targets)
        self.val_confmat(probs, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        f1 = self.val_f1.compute()  # Get current val F1
        self.val_f1_best(f1)  # Update best so far (highest) val F1
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

        #plt.figure(figsize=(10, 8))
        #confmat = self.val_confmat.compute().cpu().numpy()
        #print("Confusion Matrix:\n", confmat)
        #sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues')
        #plt.xlabel('Predicted labels')
        #plt.ylabel('True labels')
        #plt.title('Confusion Matrix')

        #self.logger.experiment.log_figure(
        #    "val/confusion_matrix", plt.gcf(), self.current_epoch
        #)
        #plt.close()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, probs, targets = self.model_step(batch)

        # Update and log metrics
        self.test_loss(loss)
        self.test_acc(probs, targets)
        self.test_f1(probs, targets)
        self.test_precision(probs, targets)
        self.test_recall(probs, targets)
        self.test_confmat(probs, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        # Get and print the final test confusion matrix
        confmat = self.test_confmat.compute().cpu().numpy()
        print("Test Confusion Matrix:")
        print(confmat)

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
        Returns predictions as class indices (0 to num_classes-1)
        """
        ids, features = batch
        logits = self(features)  # -> Tensor [B, num_classes]
        probs = torch.softmax(logits, dim=1)
        pred_classes = torch.argmax(probs, dim=1)  # Get predicted class indices
        return {
            "ids": ids.cpu().numpy(),
            "preds": pred_classes.cpu().numpy(),
            "probs": probs.cpu().numpy()  # Return probabilities for all classes
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
        return super(MulticlassClassificationModule, cls).load_from_checkpoint(
            checkpoint_path, map_location=map_location, **kwargs
        )
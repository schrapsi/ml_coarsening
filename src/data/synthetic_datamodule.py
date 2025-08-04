from lightning import LightningDataModule
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler


class SyntheticDataModule(LightningDataModule):
    def __init__(
            self,
            n_samples=1000,
            n_features=20,
            noise_level=0.05,
            relationship='linear',
            batch_size=32,
            num_workers=0,
            train_val_test_split=[0.7, 0.15, 0.15],
            scaler=None,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_level = noise_level
        self.relationship = relationship
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = train_val_test_split
        self.scaler = scaler or StandardScaler()

    def prepare_data(self):
        # Nothing to download or prepare in advance
        pass

    def setup(self, stage=None):
        # Generate synthetic data
        X = np.random.randn(self.n_samples, self.n_features)

        # Create target with known relationship
        if self.relationship == 'linear':
            weights = np.zeros(self.n_features)
            weights[:3] = [0.5, -0.3, 0.7]
            y = np.dot(X, weights)
        elif self.relationship == 'nonlinear':
            y = 0.3 * X[:, 0] ** 2 + 0.5 * X[:, 1] - 0.2 * X[:, 2] * X[:, 0]

        # Add noise
        y += self.noise_level * np.random.randn(self.n_samples)

        # Scale to [0, 1] range for sigmoid compatibility
        y = (y - y.min()) / (y.max() - y.min())

        # Train/val/test split
        train_size = int(self.n_samples * self.split[0])
        val_size = int(self.n_samples * self.split[1])

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Create PyTorch datasets
        self.train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        self.val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        self.test_dataset = TensorDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_feature_count(self):
        return self.n_features
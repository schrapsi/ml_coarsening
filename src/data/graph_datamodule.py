from pathlib import Path
import pandas as pd

from lightning import LightningDataModule

from src.utils.data_import import feature_matrix_n_performance

from torch.utils.data import TensorDataset, DataLoader

import torch


class GraphDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            features_file: str = None,
            graphs_file: str = None,
            batch_size: int = 32,
            num_workers: int = 0,
            train_val_test_split: list[float] = [0.7, 0.15, 0.15],
            data_amount: int = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = train_val_test_split

        if graphs_file and Path(graphs_file).exists():
            with open(graphs_file, 'r') as f:
                self.graphs = [line.strip() for line in f if line.strip()]
        else:
            FileNotFoundError(f"Graph file {graphs_file} not found")

        if features_file and Path(features_file).exists():
            with open(features_file, 'r') as f:
                self.features = [line.strip() for line in f if line.strip()]
        else:
            FileNotFoundError(f"Features file {features_file} not found")

        amount_per_graph = data_amount // len(self.graphs) if data_amount else None
        self.data_amount = amount_per_graph

    def prepare_data(self):
        for graph in self.graphs:
            full_path = Path(self.data_dir) / graph
            if not full_path.exists():
                raise FileNotFoundError(f"Graph directory {full_path} not found")
        pass

    def setup(self, stage=None):
        combined = pd.DataFrame()

        for graph in self.graphs:
            graph_path = str(Path(self.data_dir) / graph / "/")
            fm = feature_matrix_n_performance(graph_path, self.data_amount)

            # Select only specified features if provided
            if self.features:
                # Make sure to keep the target column 'frequency'
                keep_cols = self.features + ['frequency']
                keep_cols = [col for col in keep_cols if col in fm.columns]
                fm = fm[keep_cols]

            combined = pd.concat([combined, fm], axis=0, ignore_index=True)

        # Split the data
        X = combined.drop('frequency', axis=1)
        y = combined['frequency']

        # Convert to tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        # Create datasets for training, validation, and testing
        dataset = TensorDataset(X_tensor, y_tensor)

        # Calculate split sizes
        train_size = int(len(dataset) * self.split[0])
        val_size = int(len(dataset) * self.split[1])
        test_size = len(dataset) - train_size - val_size

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        # Return the train dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        # Return the test dataloader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_feature_count(self):
        # Return the number of features
        if self.features:
            return len(self.features)
        else:
            raise ValueError("Features not specified or not found in the dataset.")

if __name__ == "__main__":
    # Example usage
    _ = GraphDataModule(None, None, None, None)
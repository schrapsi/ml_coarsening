from pathlib import Path
import pandas as pd

from lightning import LightningDataModule

from src.utils.data_import import feature_matrix_n_performance, feature_matrix_n

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import torch


class GraphDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir,  # Directory containing the graph data either a str or a list of str
            features_file: str = None,
            graphs_file: str = None,
            batch_size: int = 32,
            num_workers: int = 0,
            train_val_test_split: list[float] = [0.7, 0.15, 0.15],
            data_amount: int = None,
            scaler=None,
    ):
        super().__init__()
        if isinstance(data_dir, str):
            self.data_dir = [data_dir]
        else:
            self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = train_val_test_split
        self.scaler = scaler
        self.graph_paths = []

        if graphs_file and Path(graphs_file).exists():
            with open(graphs_file, 'r') as f:
                self.graphs = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Graph file {graphs_file} not found")

        if features_file and Path(features_file).exists():
            with open(features_file, 'r') as f:
                self.features = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Features file {features_file} not found")

        amount_per_graph = data_amount // len(self.graphs) if data_amount else None
        self.data_amount = amount_per_graph

    def prepare_data(self):
        for graph in self.graphs:
            found = False
            for data_dir in self.data_dir:
                full_path = Path(data_dir) / graph
                if full_path.exists():
                    found = True
                    path = str(full_path / "") + "/"
                    if path not in self.graph_paths:
                        self.graph_paths.append(path)
                    break
            if not found:
                raise FileNotFoundError(f"Graph directory {full_path} not found")

    def setup(self, stage=None):
        combined = pd.DataFrame()

        for path in self.graph_paths:
            fm = feature_matrix_n_performance(path, self.data_amount)
            # Select only specified features if provided
            if self.features:
                # Make sure to keep the target column 'frequency'
                keep_cols = self.features + ['frequency']
                keep_cols = [col for col in keep_cols if col in fm.columns]
                fm = fm[keep_cols]

            combined = pd.concat([combined, fm], axis=0, ignore_index=True)

        print(f"Combined Feature Matrix shape: {combined.shape}")
        # 2. Split into train, val, test DataFrames
        train_frac, val_frac, test_frac = self.split
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Splits must sum to 1."

        # Check if validation or test fractions are zero
        use_train_for_val = val_frac < 1e-6
        use_train_for_test = test_frac < 1e-6

        # Normal splitting if both val and test are non-zero
        if not (use_train_for_val or use_train_for_test):
            train_df, temp_df = train_test_split(
                combined,
                test_size=(val_frac + test_frac),
                random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_frac / (val_frac + test_frac),
                random_state=42
            )
        else:
            train_df = combined
            val_df = train_df if use_train_for_val else pd.DataFrame()
            test_df = train_df if use_train_for_test else pd.DataFrame()
            if not use_train_for_val and not use_train_for_test:
                # Handle case where only one is zero
                val_test_df = combined.sample(frac=(val_frac + test_frac), random_state=42)
                if use_train_for_val:
                    test_df = val_test_df
                else:
                    val_df = val_test_df

        # 3. Separate features and target
        X_train = train_df.drop('frequency', axis=1)
        y_train = train_df['frequency']

        # 4. Fit scaler on train
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        # 5. Create train dataset
        self.train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        )

        # 6. Create val dataset (using train if needed)
        if use_train_for_val:
            self.val_dataset = self.train_dataset
        else:
            X_val = val_df.drop('frequency', axis=1)
            y_val = val_df['frequency']
            X_val_scaled = self.scaler.transform(X_val)
            self.val_dataset = TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
            )

        # 7. Create test dataset (using train if needed)
        if use_train_for_test:
            self.test_dataset = self.train_dataset
        else:
            X_test = test_df.drop('frequency', axis=1)
            y_test = test_df['frequency']
            X_test_scaled = self.scaler.transform(X_test)
            self.test_dataset = TensorDataset(
                torch.tensor(X_test_scaled, dtype=torch.float32),
                torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
            )


    def train_dataloader(self):
        # Return the train dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )


    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True
        )


    def test_dataloader(self):
        # Return the test dataloader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True
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

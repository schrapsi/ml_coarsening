from pathlib import Path
import pandas as pd
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import WeightedRandomSampler
from src.utils.data_import import feature_matrix_n_performance
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import torch


class MulticlassClassificationDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            features_file: str = None,
            graphs_file: str = None,
            batch_size: int = 32,
            num_workers: int = 0,
            train_val_test_split: list[float] = [0.7, 0.15, 0.15],
            data_amount: int = None,
            scaler=None,
            num_classes: int = 10,  # Default 10 classes (0-9) for range [0,1]
            use_smote: bool = False,
            smote_sampling_strategy: str = 'auto',
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = train_val_test_split
        self.scaler = scaler
        self.num_classes = num_classes
        self.use_smote = use_smote
        self.smote_sampling_strategy = smote_sampling_strategy

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
        self.class_amount = data_amount / num_classes if data_amount else None

    def prepare_data(self):
        for graph in self.graphs:
            full_path = Path(self.data_dir) / graph
            if not full_path.exists():
                raise FileNotFoundError(f"Graph directory {full_path} not found")

    def setup(self, stage=None):
        combined = pd.DataFrame()

        for graph in self.graphs:
            graph_path = str(Path(self.data_dir) / graph / "") + "/"
            fm = feature_matrix_n_performance(graph_path, self.data_amount)

            # Select only specified features if provided
            if self.features:
                keep_cols = self.features + ['frequency']
                keep_cols = [col for col in keep_cols if col in fm.columns]
                fm = fm[keep_cols]

            combined = pd.concat([combined, fm], axis=0, ignore_index=True)

        # Convert frequency to multiclass labels (0-9)
        # Clip values to ensure they're in [0,1] range
        frequencies = combined['frequency']

        # Convert to classes 0-9
        # Class 0 is [0, 0.1), Class 1 is [0.1, 0.2), ..., Class 9 is [0.9, 1.0]
        multiclass_labels = (frequencies * self.num_classes).astype(int)
        # Handle the edge case where frequency=1.0 (should be class 9, not 10)
        multiclass_labels = np.minimum(multiclass_labels, self.num_classes - 1)

        # Calculate class distribution
        class_counts = multiclass_labels.value_counts().sort_index()
        total_count = len(multiclass_labels)

        # Print class distribution statistics
        print(f"Class distribution across {self.num_classes} classes:")
        for class_idx, count in class_counts.items():
            lower_bound = class_idx * 0.1
            upper_bound = (class_idx + 1) * 0.1
            print(f"  Class {class_idx} [{lower_bound:.1f}-{upper_bound:.1f}): {count} ({count / total_count:.2%})")

        # Split into train, val, test DataFrames
        train_frac, val_frac, test_frac = self.split
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Splits must sum to 1."

        X = combined.drop('frequency', axis=1)
        train_idx, temp_idx = train_test_split(
            range(len(X)),
            test_size=(val_frac + test_frac),
            random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_frac / (val_frac + test_frac),
            random_state=42
        )

        # Separate features and multiclass targets
        X_train = X.iloc[train_idx]
        y_train = multiclass_labels.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = multiclass_labels.iloc[val_idx]
        X_test = X.iloc[test_idx]
        y_test = multiclass_labels.iloc[test_idx]

        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        if self.use_smote:
            print("Applying SMOTE oversampling to training data...")
            print(self.get_min_class_samples(y_train))
            smote = SMOTE(sampling_strategy=self.smote_sampling_strategy,
                          k_neighbors=min(5, self.get_min_class_samples(y_train)),
                          random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

            majority_indices = np.where(y_train_resampled == 0)[0]
            if len(majority_indices) > self.class_amount:
                keep_indices = np.random.choice(majority_indices, self.class_amount, replace=False)
                other_indices = np.where(y_train_resampled != 0)[0]
                final_indices = np.concatenate([keep_indices, other_indices])
                X_train_resampled = X_train_resampled[final_indices]
                y_train_resampled = y_train_resampled[final_indices]

            # Print class distribution after SMOTE
            class_counts_after = np.bincount(y_train_resampled, minlength=self.num_classes)
            print("Class distribution after SMOTE:")
            for class_idx, count in enumerate(class_counts_after):
                lower_bound = class_idx * 0.1
                upper_bound = (class_idx + 1) * 0.1
                print(
                    f"  Class {class_idx} [{lower_bound:.1f}-{upper_bound:.1f}): {count} ({count / len(y_train_resampled):.2%})")

            # Create tensor dataset with SMOTE-resampled data
            self.train_dataset = TensorDataset(
                torch.tensor(X_train_resampled, dtype=torch.float32),
                torch.tensor(y_train_resampled, dtype=torch.long)
            )

        else:
            # Original approach without SMOTE
            self.train_dataset = TensorDataset(
                torch.tensor(X_train_scaled, dtype=torch.float32),
                torch.tensor(y_train.values, dtype=torch.long)
            )
        self.val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.long)
        )
        self.test_dataset = TensorDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.long)
        )

    def get_min_class_samples(self, y):
        """Return the count of samples in the smallest class"""
        class_counts = np.bincount(y, minlength=self.num_classes)
        return max(1, min([count for count in class_counts if count > 0]))

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
        if self.features:
            return len(self.features)
        else:
            raise ValueError("Features not specified or not found in the dataset.")
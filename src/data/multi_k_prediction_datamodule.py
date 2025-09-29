from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader

from src.utils.data_import import feature_matrix_n_performance


class MultiKPredictionDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 graphs_file: str = None,
                 k_values: list = None,
                 num_workers: int = 0,
                 scaler=None,
                 features=None,
                 ):
        super().__init__()
        self.graph_path_mapping = {}
        self._predict_datasets = None
        if isinstance(data_dir, str):
            self.data_dir = [data_dir]
        else:
            self.data_dir = data_dir
        self.num_workers = num_workers
        self.scaler = scaler
        self.features = features if features else []
        self.k_values = k_values if k_values else [8]  # Default k value

        if graphs_file and Path(graphs_file).exists():
            with open(graphs_file, 'r') as f:
                self.graphs = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Graph file {graphs_file} not found")

    def prepare_data(self):
        print("Preparing data for multi-k graph prediction...")
        print(f"Using k values: {self.k_values}")
        print(f"Processing graphs: {self.graphs}")
        for graph in self.graphs:
            print(f"Preparing data for graph: {graph}")
            found = False
            for data_dir in self.data_dir:
                full_path = Path(data_dir) / graph
                if full_path.exists():
                    found = True
                    path = str(full_path / "") + "/"
                    self.graph_path_mapping[graph] = path
                    print(f"Found graph {graph} at {path}")
                    break
            if not found:
                raise FileNotFoundError(f"Graph directory for {graph} not found")

    def setup(self, stage=None):
        # Create datasets for each graph
        self._predict_datasets = {}
        for graph, path in self.graph_path_mapping.items():
            # Create base feature matrix for the graph (without labels)
            fm = feature_matrix_n_performance(path, with_id=True)
            ids = fm[["id_high_degree", "id_low_degree"]].to_numpy(dtype=int)

            # Get features only (no labels)
            feats = fm.drop(columns=["id_high_degree", "id_low_degree", fm.columns[-1]])

            # For each k value, create a dataset with k as additional feature
            self._predict_datasets[graph] = {}
            for k in self.k_values:
                print(f"Setting up data for graph {graph}, k={k}")

                new_feats = feats.copy()
                new_feats['k_value'] = k
                new_feats = new_feats[self.features]

                if self.scaler:
                    new_feats = self.scaler.transform(new_feats)

                self._predict_datasets[graph][k] = TensorDataset(
                    torch.tensor(ids, dtype=torch.int64),
                    torch.tensor(new_feats, dtype=torch.float32)
                )

    def predict_dataloader(self):
        # Return a nested dictionary of DataLoaders for each graph and k value
        dataloaders = {}
        for graph, k_datasets in self._predict_datasets.items():
            dataloaders[graph] = {}
            for k, dataset in k_datasets.items():
                dataloaders[graph][k] = DataLoader(dataset, batch_size=len(dataset), num_workers=self.num_workers)
        return dataloaders

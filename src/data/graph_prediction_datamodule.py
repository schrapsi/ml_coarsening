from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset

from src.utils.data_import import feature_matrix_n_performance
from torch.utils.data import DataLoader


class GraphPredictionDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 graphs_file: str = None,
                 num_workers: int = 0,
                 scaler=None,
                 features=None,
                 ):
        super().__init__()
        self.graph_path_mapping = None
        self._predict_datasets = None
        if isinstance(data_dir, str):
            self.data_dir = [data_dir]
        else:
            self.data_dir = data_dir
        self.num_workers = num_workers
        self.scaler = scaler
        self.features = features if features else []

        if graphs_file and Path(graphs_file).exists():
            with open(graphs_file, 'r') as f:
                self.graphs = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Graph file {graphs_file} not found")

    def prepare_data(self):
        self.graph_path_mapping = {}

        for graph in self.graphs:
            found = False
            for data_dir in self.data_dir:
                full_path = Path(data_dir) / graph
                if full_path.exists():
                    found = True
                    path = str(full_path / "") + "/"
                    self.graph_path_mapping[graph] = path
                    break
            if not found:
                raise FileNotFoundError(f"Graph directory {path} not found")

    def setup(self, stage=None):
        # Go through all graphs in self.graphs and create a DataLoader dictionary with DataLoader
        # for each graph containing the features and labels from the whole graph
        self._predict_datasets = {}
        for graph, path in self.graph_path_mapping.items():
            fm = feature_matrix_n_performance(path, with_id=True)
            ids = fm[["id_high_degree", "id_low_degree"]].to_numpy(dtype=int)
            labels = fm[fm.columns[-1]].to_numpy(dtype=float)
            feats = fm.drop(columns=["id_high_degree", "id_low_degree", fm.columns[-1]])
            feats = feats[self.features]
            if self.scaler:
                feats = self.scaler.transform(feats)

            self._predict_datasets[graph] = TensorDataset(
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor(feats, dtype=torch.float32),
            )

    def predict_dataloader(self):
        # Return a DataLoader for each graph in self._predict_datasets
        return {graph: DataLoader(dataset, batch_size=len(dataset), num_workers=self.num_workers)
                for graph, dataset in self._predict_datasets.items()}

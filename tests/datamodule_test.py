# tests/test_your_datamodule.py
from pathlib import Path
import pytest
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

from src.data.graph_datamodule import GraphDataModule

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def data_dir() -> str:
    """Returns the path to the test data directory."""
    return str(PROJECT_ROOT / "tests/test_data")


@pytest.fixture
def graphs_file() -> str:
    """Returns the path to the test graphs file."""
    return str(PROJECT_ROOT / "configs/data/graphs/test_set.txt")


@pytest.mark.parametrize("batch_size", [32, 128])
@pytest.mark.parametrize("features_file_name", ["all.txt", "cost_0.txt"])
@pytest.mark.parametrize("scaler", [StandardScaler(), MinMaxScaler()])
@pytest.mark.parametrize("split", [[0.7, 0.15, 0.15], [1.0, 0.0, 0.0]])
def test_graph_datamodule_integration(
        batch_size: int,
        features_file_name: str,
        scaler,
        split: list[float],
        data_dir: str,
        graphs_file: str,
) -> None:
    """Comprehensive integration test for GraphDataModule."""
    features_file = str(PROJECT_ROOT / "configs/data/features" / features_file_name)
    data_amount = 1000
    train_frac, val_frac, test_frac = split

    dm = GraphDataModule(
        data_dir=data_dir,
        graphs_file=graphs_file,
        features_file=features_file,
        batch_size=batch_size,
        num_workers=0,
        train_val_test_split=split,
        data_amount=data_amount,
        scaler=scaler,
    )

    dm.prepare_data()
    dm.setup()

    # Test setup correctness
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None

    # Test dataloader creation
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


    # Test batch properties
    x, y = next(iter(train_loader))
    assert len(x) <= batch_size
    assert len(y) <= batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32

    # Test feature count
    num_features = dm.get_feature_count()
    assert x.shape[1] == num_features

    with open(features_file, "r") as f:
        expected_num_features = len([line for line in f if line.strip()])
    assert num_features == expected_num_features

    # Test scaling
    if isinstance(scaler, StandardScaler):
        # Mean should be close to 0 and std close to 1 for training data
        assert torch.allclose(x.mean(), torch.tensor(0.0), atol=0.5)
        assert torch.allclose(x.std(), torch.tensor(1.0), atol=0.5)
    elif isinstance(scaler, MinMaxScaler):
        # Values should be between 0 and 1
        assert torch.all(x >= 0) and torch.all(x <= 1)

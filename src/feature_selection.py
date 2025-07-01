# src/feature_selection.py
import logging

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="feature_selection.yaml")
def main(cfg: DictConfig) -> None:
    """
    Runs feature selection using a RandomForestRegressor.

    This script uses your existing datamodule to load data, trains a
    RandomForestRegressor, ranks features by importance, and saves the
    top N features to a new file.
    """
    log.info("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    log.info("Loading and concatenating training data...")
    # This assumes your dataloader yields batches with 'x' (features) and 'y' (target)
    # You might need to adjust this based on your specific DataModule implementation
    train_dataloader = datamodule.train_dataloader()
    print(train_dataloader)
    X_list = train_dataloader.dataset.tensors[0]
    y_list = train_dataloader.dataset.tensors[1]

    # Convert to pandas DataFrame/Series for scikit-learn
    X = pd.DataFrame(X_list.numpy())
    y = pd.Series(y_list.numpy().squeeze())

    # Load feature names
    X.columns = datamodule.features

    log.info(f"Training RandomForestRegressor on {len(X)} samples...")
    model = RandomForestRegressor(**cfg.model)
    model.fit(X, y)

    log.info("Ranking features by importance...")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    ranked_features = importances.sort_values(ascending=False)

    log.info(f"Top 20 features:\n{ranked_features.head(20)}")

    # Save the top N features to a new file
    top_n = cfg.get("top_n", 50)
    output_path = cfg.get("output_path", f"configs/data/features/top_{top_n}_features.txt")

    log.info(f"Saving top {top_n} features to '{output_path}'...")
    ranked_features.head(top_n).index.to_series().to_csv(
        output_path, index=False, header=False
    )
    log.info("Feature selection finished.")


if __name__ == "__main__":
    main()
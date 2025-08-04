# ML Coarsening: Machine Learning-Guided Graph Partitioning

This project implements machine learning techniques to improve coarsening for multilevel graph partitioning.

## 🎯 Project Goal

The primary goal is to enhance balanced graph partitioning by using machine learning to guide the coarsening phase of the multilevel graph partitioner Mt-KaHyPar. Traditional coarsening methods (e.g., label-propagation clustering) rely only on local connectivity and can destroy global structure, limiting final partition quality. This project addresses this limitation by training deep neural networks to predict edge-cut probabilities and to guide vertex clustering decisions with those predictions.


## 🚀 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ml_coarsening
```

2. **Install dependencies using uv:**
```bash
uv sync
```

3. **Activate environment (if using virtual environment):**
```bash
source .venv/bin/activate
```

## 🔧 Usage

### Basic Training Commands

#### Multi-Class Classification (Default)
```bash
uv run -m src.train experiment=multi_classification
```

#### Binary Classification
```bash
uv run -m src.train experiment=bin_classification
```

#### Regression with BCE Loss
```bash
uv run -m src.train experiment=regression_bce
```

### Advanced Training Options

#### Custom Configuration Override
```bash
uv run -m src.train experiment=multi_classification \
  model.focal_gamma=4.0 \
  trainer=gpu \
  data.graphs_file=configs/data/graphs/custom.txt
```

#### CPU Training
```bash
uv run -m src.train experiment=multi_classification trainer=cpu
```

#### Debug Mode
```bash
uv run -m src.train experiment=multi_classification debug=overfit
```

### Inference

#### Run Inference on Trained Model
```bash
uv run -m src.inference 
```

### Available Experiment Configurations

| Experiment | Description | Configuration File |
|------------|-------------|-------------------|
| `multi_classification` | Multi-class edge cut prediction | `configs/experiment/multi_classification.yaml` |
| `bin_classification` | Binary edge cut prediction | `configs/experiment/bin_classification.yaml` |
| `regression_bce` | Regression with BCE loss | `configs/experiment/regression_bce.yaml` |
| `mss_1_20` | Specific dataset configuration | `configs/experiment/mss_1_20.yaml` |

### Model Types

- **Binary Classification**: Predicts whether an edge will be cut (binary outcome)
- **Multi-Class Classification**: Predicts cut probability in discrete bins
- **Regression**: Directly predicts continuous cut probability values

## 📊 Experiment Tracking

The project uses **MLflow** for experiment tracking. To view results:

```bash
mlflow ui
```

Navigate to the provided URL to view:
- Training metrics and validation scores
- Model parameters and hyperparameters
- Training artifacts and model checkpoints


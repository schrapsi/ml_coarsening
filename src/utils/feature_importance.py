import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import shap
from pathlib import Path

from src.models.ml_coarsening_module import MLCoarseningModule
from src.utils.data_import import feature_matrix_n_performance


# 1. Load and prepare data from the problematic graphs
def load_graph_data(data_dir, graph_names, features_list, amount_per_graph=None):
    combined = pd.DataFrame()
    for graph in graph_names:
        graph_path = str(Path(data_dir) / graph / "") + "/"
        fm = feature_matrix_n_performance(graph_path, amount_per_graph)

        # Select only specified features
        if features_list:
            keep_cols = features_list + ['frequency']
            keep_cols = [col for col in keep_cols if col in fm.columns]
            fm = fm[keep_cols]

        combined = pd.concat([combined, fm], axis=0, ignore_index=True)

    return combined


# 2. Correlation analysis
def correlation_analysis(df):
    corr = df.corr()
    target_corr = corr['frequency'].sort_values(ascending=False)

    # Plot correlations
    plt.figure(figsize=(10, 8))
    plt.barh(target_corr.index[1:], target_corr.values[1:])
    plt.title('Feature Correlation with Target')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')

    return target_corr


# 3. Permutation importance
def permutation_importance_analysis(model, X, y, feature_names):
    # Convert model to evaluation mode
    model.eval()

    def model_predict(X_array):
        X_tensor = torch.tensor(X_array, dtype=torch.float32)
        with torch.no_grad():
            return model(X_tensor).cpu().numpy().flatten()

    # Calculate permutation importance
    result = permutation_importance(
        estimator=model_predict,
        X=X,
        y=y,
        n_repeats=10,
        random_state=42
    )

    # Sort features by importance
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })
    importances = importances.sort_values('Importance', ascending=False)

    # Plot results
    plt.figure(figsize=(10, 8))
    plt.barh(importances['Feature'], importances['Importance'])
    plt.title('Permutation Feature Importance')
    plt.xlabel('Mean decrease in model performance')
    plt.tight_layout()
    plt.savefig('permutation_importance.png')

    return importances


# 4. SHAP analysis
def shap_analysis(model, X, feature_names, n_samples=500):
    # Sample data for SHAP analysis (for speed)
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    X_tensor = torch.tensor(X_sample, dtype=torch.float32)

    # Create explainer
    explainer = shap.DeepExplainer(model, X_tensor)
    shap_values = explainer.shap_values(X_tensor)

    # Plot SHAP summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')

    return shap_values


# Main execution function
def analyze_feature_importance(data_dir, graph_names, features_file, model_path):
    # Load feature names
    with open(features_file, 'r') as f:
        features = [line.strip() for line in f if line.strip()]

    # Load data
    data = load_graph_data(data_dir, graph_names, features)
    X = data.drop('frequency', axis=1)
    y = data['frequency']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load model
    model = MLCoarseningModule.load_from_checkpoint(model_path, map_location=torch.device('cpu'))

    print("Running correlation analysis...")
    corr_results = correlation_analysis(data)
    print(f"Top 10 correlated features:\n{corr_results.head(10)}")

    #print("\nRunning permutation importance...")
    #perm_results = permutation_importance_analysis(model, X_scaled, y.values, X.columns)
    #print(f"Top 10 important features:\n{perm_results.head(10)}")

    print("\nRunning SHAP analysis...")
    shap_values = shap_analysis(model, X_scaled, X.columns)

    print("\nAnalysis complete. Results saved as images.")

    return {
        'correlation': corr_results,
        'permutation': perm_results,
        'shap_values': shap_values
    }
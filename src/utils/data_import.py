import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame


def feature_matrix_n_performance(path, amount=None, with_id=False, balanced=False) -> DataFrame:
    print("Starting to Parse: ", path)

    # 1. Define datatypes for faster CSV reading
    node_dtypes = {'node_id': 'int32'}
    edge_dtypes = {'id_high_degree': 'int32', 'id_low_degree': 'int32'}
    label_dtypes = {'id_high_degree': 'int32', 'id_low_degree': 'int32', 'frequency': 'float32'}

    # 2. Read max value from header to avoid reading freq_all.csv twice
    with open(path + "freq_all.csv", 'r') as f:
        _ = f.readline()
        second_line = f.readline()
    max_value = int(re.search(r"# max=(\d+)", second_line).group(1))

    # 3. Read data with optimized parameters
    global_f = pd.read_csv(path + "global.csv")

    if amount is None:
        edges = pd.read_csv(path + "edges_shuf.csv", dtype=edge_dtypes)
    else:
        edges = pd.read_csv(path + "edges_shuf.csv", nrows=int(amount), dtype=edge_dtypes)

    # 4. Create indexes before merging for better performance
    nodes = pd.read_csv(path + "nodes.csv", dtype=node_dtypes)
    nodes.set_index('node_id', inplace=True)

    # 5. Read labels
    label_file = "freq_balanced.csv" if balanced else "freq_all.csv"
    labels = pd.read_csv(path + label_file, comment='#', dtype=label_dtypes)

    # 6. More efficient way to replicate global features
    row_dict = global_f.iloc[0].to_dict()
    replicated_global_f = pd.DataFrame([row_dict] * len(edges))

    # 7. Faster node feature lookups with pre-indexed dataframe
    # Extract high and low degree IDs
    high_ids = edges['id_high_degree'].values
    low_ids = edges['id_low_degree'].values

    # Direct lookup instead of merge
    node1_features = nodes.loc[high_ids].reset_index(drop=True)
    node2_features = nodes.loc[low_ids].reset_index(drop=True)

    # 8. Combine features - keep IDs from edges dataframe
    id_columns = edges[['id_high_degree', 'id_low_degree']]

    # Combine features with IDs in correct position
    combined_features = pd.concat([replicated_global_f, edges.drop(columns=['id_high_degree', 'id_low_degree']),
                                   node1_features, node2_features], axis=1)

    # 9. Create a dictionary for label lookup (faster than merge)
    label_dict = dict(zip(zip(labels['id_high_degree'], labels['id_low_degree']), labels['frequency']))

    # 10. Get frequencies using the dictionary
    edge_pairs = list(zip(edges['id_high_degree'], edges['id_low_degree']))
    frequencies = [label_dict.get(pair, 0) for pair in edge_pairs]

    # Name the frequency series properly
    sorted_labels = pd.Series(frequencies, name='frequency') / max_value

    # 11. Handle ID columns based on with_id parameter
    if with_id:
        # Put ID columns at the beginning of the DataFrame
        combined_features = pd.concat([id_columns, combined_features], axis=1)

    # 12. Handle missing values
    na_cols = combined_features.columns[combined_features.isna().any()].tolist()
    if na_cols:
        last_folder = Path(path).parts[-1]
        print(f"Missing values in {last_folder} for columns: {na_cols}")
    combined_features.fillna(1, inplace=True)

    # 13. Combine features and labels
    fm = pd.concat([combined_features, sorted_labels], axis=1)

    # 14. More efficient duplicate column handling
    if fm.columns.duplicated().any():
        cols = pd.Series(fm.columns)
        for dup in fm.columns[fm.columns.duplicated(keep=False)].unique():
            cols[fm.columns == dup] = [f"{dup}.{i}" if i > 0 else dup
                                       for i in range(sum(fm.columns == dup))]
        fm.columns = cols

    return fm
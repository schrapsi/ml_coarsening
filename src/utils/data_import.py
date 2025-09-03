import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame


def feature_matrix_n(path, amount=None, with_id=False, balanced=False) -> DataFrame:
    print("Starting to Parse: ", path)
    global_f = pd.read_csv(path + "global.csv")
    if amount is None:
        edges = pd.read_csv(path + "edges_shuf.csv")
    else:
        edges = pd.read_csv(path + "edges_shuf.csv", nrows=int(amount))

    nodes = pd.read_csv(path + "nodes.csv")
    if balanced:
        labels = pd.read_csv(path + "freq_balanced.csv", comment='#')
    else:
        labels = pd.read_csv(path + "freq_all.csv", comment='#')

    row = global_f.iloc[0]
    replicated_global_f = pd.concat([row.to_frame().T] * len(edges), ignore_index=True)
    node1_features = pd.merge(edges[['id_high_degree']], nodes, left_on='id_high_degree', right_on='node_id',
                              how='left')
    node2_features = pd.merge(edges[['id_low_degree']], nodes, left_on='id_low_degree', right_on='node_id',
                              how='left')

    node1_features = node1_features.drop(columns=['id_high_degree', 'node_id'])
    node2_features = node2_features.drop(columns=['id_low_degree', 'node_id'])

    combined_features = pd.concat([replicated_global_f, edges, node1_features, node2_features], axis=1)

    max_row = pd.read_csv(path + "freq_all.csv", low_memory=False, nrows=2)
    max_row = max_row.iloc[0, 0]
    max_value = int(re.search(r"# max=(\d+)", max_row).group(1))

    labels['id_high_degree'] = labels['id_high_degree'].astype(int)
    labels['id_low_degree'] = labels['id_low_degree'].astype(int)

    merged_df = pd.merge(combined_features[['id_high_degree', 'id_low_degree']], labels,
                         on=['id_high_degree', 'id_low_degree'], how='left')

    sorted_labels = merged_df['frequency']
    if with_id:
        combined_features = pd.concat([combined_features[['id_high_degree', 'id_low_degree']],
                                       combined_features.drop(columns=['id_high_degree', 'id_low_degree'])], axis=1)
    else:
        combined_features = combined_features.drop(columns=['id_high_degree', 'id_low_degree'])

    sorted_labels = sorted_labels / max_value

    na_series = combined_features.isna().sum()
    na_list = na_series.tolist()

    last_folder = Path(path).parts[-1]
    if sum(na_list) > 0:
        print(f"Missing values in {last_folder}")

    combined_features.fillna(1, inplace=True)

    fm = pd.concat([combined_features, sorted_labels], axis=1)
    cols = pd.Series(fm.columns)
    for duplicate in fm.columns[fm.columns.duplicated(keep=False)].unique():
        names = []
        for n in range(sum(fm.columns == duplicate)):
            if n == 0:
                names.append(f'{duplicate}')
            else:
                names.append(f'{duplicate}.{n}')
        cols[fm.columns == duplicate] = names
    fm.columns = cols

    return fm


def feature_matrix_n_performance(path, amount=None, with_id=False, balanced=False) -> DataFrame:
    print("Starting to Parse: ", path)

    #TODO Change default back to freq_all.csv
    label_file = "freq_balanced.csv" if balanced else "freqk4.csv"
    # 1. Define datatypes for faster CSV reading
    node_dtypes = {'node_id': 'int32'}
    edge_dtypes = {'id_high_degree': 'int32', 'id_low_degree': 'int32'}
    label_dtypes = {'id_high_degree': 'int32', 'id_low_degree': 'int32', 'frequency': 'float32'}

    with open(path + label_file, 'r') as f:
        first_line = f.readline()
        second_line = f.readline()
    try:
        max_value = int(re.search(r"# max=(\d+)", first_line).group(1))
    except AttributeError:
        try:
            max_value = int(re.search(r"# max=(\d+)", second_line).group(1))
        except AttributeError:
            raise ValueError(f"No max value found in {path+label_file}")

    # 3. Read data with optimized parameters
    global_f = pd.read_csv(path + "global.csv")

    if Path.exists(Path(path + "edges_shuf.csv")):
        edge_file = path + "edges_shuf.csv"
    else:
        edge_file = path + "edges.csv" # Fallback to edges.csv if edges_shuf.csv does not exist

    if amount is None:
        edges = pd.read_csv(edge_file, dtype=edge_dtypes)
    else:
        edges = pd.read_csv(edge_file, nrows=int(amount), dtype=edge_dtypes)

    # 4. Create indexes before merging for better performance
    nodes = pd.read_csv(path + "nodes.csv", dtype=node_dtypes)
    nodes.set_index('node_id', inplace=True)

    # 5. Read labels

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


def feature_matrix_multi_k(path, amount=None, with_id=False) -> DataFrame:
    """
    Creates a feature matrix using all freqk[number].csv files found in the directory.

    Args:
        path: Path to the graph folder
        amount: Number of edges to process (None for all)
        with_id: Whether to include node IDs in the feature matrix

    Returns:
        Combined feature matrix with data from all k-values
    """
    print(f"Processing all k-values from: {path}")
    all_data_frames = []

    # 1. Scrape all freqk[number].csv files in the directory
    freqk_files = []
    k_values = []
    for file in Path(path).glob("freqk*.csv"):
        match = re.match(r"freqk(\d+)\.csv", file.name)
        if match:
            k = int(match.group(1))
            freqk_files.append((k, file))
            k_values.append(k)
    if not freqk_files:
        print("No freqk[number].csv files found in the directory.")
        return pd.DataFrame()
    freqk_files.sort()  # Sort by k

    # 2. Read data that's common for all k-values once
    node_dtypes = {'node_id': 'int32'}
    edge_dtypes = {'id_high_degree': 'int32', 'id_low_degree': 'int32'}
    label_dtypes = {'id_high_degree': 'int32', 'id_low_degree': 'int32', 'frequency': 'float32'}

    if amount is None:
        edges = pd.read_csv(path + "edges_shuf.csv", dtype=edge_dtypes)
    else:
        amount = amount / len(freqk_files)
        edges = pd.read_csv(path + "edges_shuf.csv", nrows=int(amount), dtype=edge_dtypes)

    nodes = pd.read_csv(path + "nodes.csv", dtype=node_dtypes)
    nodes.set_index('node_id', inplace=True)

    global_f = pd.read_csv(path + "global.csv")
    row_dict = global_f.iloc[0].to_dict()
    replicated_global_f = pd.DataFrame([row_dict] * len(edges))

    high_ids = edges['id_high_degree'].values
    low_ids = edges['id_low_degree'].values
    node1_features = nodes.loc[high_ids].reset_index(drop=True)
    node2_features = nodes.loc[low_ids].reset_index(drop=True)

    id_columns = edges[['id_high_degree', 'id_low_degree']]
    base_features = pd.concat([
        replicated_global_f,
        edges.drop(columns=['id_high_degree', 'id_low_degree']),
        node1_features,
        node2_features
    ], axis=1)

    # 3. Process each detected k-value
    for k, freq_file_path in freqk_files:
        freq_file = str(freq_file_path)
        # Get max value from the file header
        with open(freq_file, 'r') as f:
            first_line = f.readline()
            second_line = f.readline()
        try:
            max_value = int(re.search(r"# max=(\d+)", first_line).group(1))
        except AttributeError:
            try:
                max_value = int(re.search(r"# max=(\d+)", second_line).group(1))
            except AttributeError:
                raise ValueError(f"No max value found in {freq_file}")

        # Read frequency file for current k
        labels = pd.read_csv(freq_file, comment='#', dtype=label_dtypes)

        # Create a dictionary for label lookup
        label_dict = dict(zip(zip(labels['id_high_degree'], labels['id_low_degree']), labels['frequency']))

        # Get frequencies using the dictionary
        edge_pairs = list(zip(edges['id_high_degree'], edges['id_low_degree']))
        frequencies = [label_dict.get(pair, 0) for pair in edge_pairs]

        # Create normalized frequencies
        sorted_labels = pd.Series(frequencies, name='frequency') / max_value

        # Create a copy of base features and add k as a feature
        k_features = base_features.copy()
        k_features['k_value'] = k

        # Handle ID columns based on with_id parameter
        if with_id:
            k_features = pd.concat([id_columns, k_features], axis=1)

        # Handle missing values
        na_cols = k_features.columns[k_features.isna().any()].tolist()
        if na_cols:
            last_folder = Path(path).parts[-1]
            print(f"Missing values in {last_folder} for k={k}, columns: {na_cols}")
        k_features.fillna(1, inplace=True)

        # Combine features and labels for this k
        k_data = pd.concat([k_features, sorted_labels], axis=1)
        all_data_frames.append(k_data)

    # 4. Combine all k-value data
    combined_data = pd.concat(all_data_frames, ignore_index=True)

    # 5. Handle duplicate columns efficiently
    if combined_data.columns.duplicated().any():
        cols = pd.Series(combined_data.columns)
        for dup in combined_data.columns[combined_data.columns.duplicated(keep=False)].unique():
            cols[combined_data.columns == dup] = [f"{dup}.{i}" if i > 0 else dup
                                                  for i in range(sum(combined_data.columns == dup))]
        combined_data.columns = cols

    return combined_data
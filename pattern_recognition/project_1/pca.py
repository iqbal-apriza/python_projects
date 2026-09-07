import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def main():
    dataset_dir = BASE_DIR / "dataset_example.csv"
    df = pd.read_csv(dataset_dir)
    df_norm = df.drop(columns="Class")

    N_COMP = 2

    dataset_arr = df_norm.to_numpy()
    data_rows = dataset_arr.shape[0]
    data_cols = dataset_arr.shape[1]

    global_mean = np.average(dataset_arr, axis=0)
    zero_mean = dataset_arr - global_mean
    covariance = (1 / data_rows) * zero_mean.T @ zero_mean

    eig_val, eig_vec = np.linalg.eig(covariance)

    sorted_indices = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_indices]
    eig_vec = eig_vec[:, sorted_indices]

    components = eig_vec[:, :N_COMP]
    total_var = np.sum(eig_val)

    final_data = np.dot(zero_mean, components)

    print(dataset_arr)
    print("")
    print(final_data)


if __name__ == '__main__':
    main()
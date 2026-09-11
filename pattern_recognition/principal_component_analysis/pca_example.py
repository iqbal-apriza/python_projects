import numpy as np
import pandas as pd
import argparse

from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent
DATASET_FILE    = "dataset_example.csv"
N_COMP          = 2


def fit_pca(data, n_comp):
    data_rows = data.shape[0]

    global_mean = np.average(data, axis=0)
    zero_mean = data - global_mean
    covariance = (1 / data_rows) * zero_mean.T @ zero_mean

    eig_val, eig_vec = np.linalg.eig(covariance)

    sorted_indices = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorted_indices]
    eig_vec = eig_vec[:, sorted_indices]

    components = eig_vec[:, :n_comp]
    return np.dot(zero_mean, components)
    

def main():
    dataset_dir = BASE_DIR / DATASET_FILE
    df = pd.read_csv(dataset_dir)
    df_norm = df.iloc[:, :-1]

    dataset_arr = df_norm.to_numpy()
    results = fit_pca(dataset_arr, N_COMP)

    df_res = pd.DataFrame()
    for i in range(results.shape[1]):
        df_res[f"Comp {i+1}"] = np.round(results[:,i], 2)

    df_res["Class"] = df["Class"]

    print("Original Data\n"
          "--------------------\n"
          f"{df}")

    print("\n\nReduced Data\n"
          "--------------------\n"
          f"{df_res}")


if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd

from pathlib import Path
import pca_example as pca

BASE_DIR        = Path(__file__).resolve().parent
DATASET_FILE    = "student_performance_data.csv"
N_COMP          = 5

gender_map = {
    "Female"    : 1,
    "Male"      : 2
}

major_map = {
    "Arts"          : 1,
    "Business"      : 2,
    "Education"     : 3,
    "Science"       : 4,
    "Engineering"   : 5
}

part_time_map = {
    "No"    : 1,
    "Yes"   : 2
}

excurr_map = {
    "No"    : 1,
    "Yes"   : 2
}


def main():
    dataset_dir = BASE_DIR / DATASET_FILE
    df = pd.read_csv(dataset_dir)

    df_norm = df.drop(columns="StudentID")
    df_norm["Gender"] = df_norm["Gender"].map(gender_map)
    df_norm["Major"] = df_norm["Major"].map(major_map)
    df_norm["PartTimeJob"] = df_norm["PartTimeJob"].map(part_time_map)
    df_norm["ExtraCurricularActivities"] = df_norm["ExtraCurricularActivities"].map(excurr_map)

    dataset_arr = df_norm.to_numpy()
    results = pca.fit_pca(dataset_arr, N_COMP)

    df_res = pd.DataFrame()
    for i in range(results.shape[1]):
        df_res[f"Comp {i+1}"] = np.round(results[:,i], 2)

    print("Original Data\n"
            "--------------------\n"
            f"{df}")

    print("\n\nReduced Data\n"
            "--------------------\n"
            f"{df_res}")


if __name__ == '__main__':
    main()
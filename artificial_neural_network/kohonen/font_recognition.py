import numpy as np
import argparse

import yaml
from pathlib import Path

import kohonen as kh

BASE_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset file in yaml")
    ap.add_argument("--max-epoch", default=100, type=int)
    ap.add_argument("--alpha", default=0.1, type=float)
    ap.add_argument("--decrease-fct", default=1.0, type=float)

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.dataset
    yaml_data = kh.read_yaml(yaml_dir)

    dataset = np.array([
        pattern
        for font in yaml_data["input"].values()
        for pattern in font.values()
    ])

    font_group = np.array([
        list(font.values())
        for font in yaml_data["input"].values()
    ])

    letter_group = font_group.transpose(1, 0, 2)

    input_row = dataset.shape[0]
    input_col = dataset.shape[1]

    net = kh.Kohonen(7, input_col, args.alpha, args.decrease_fct)

    for epoch in range(args.max_epoch):
        print(f"Executing epoch: {epoch+1}")

        for i in range(input_row):
            net.predict(dataset[i])
            net.update()

    # Predictions
    print("\n\n===== RESULTS =====")
    for i in range(letter_group.shape[0]):
        print(f"--- Letter {i} ---")

        for j in range(letter_group.shape[1]):
            output = net.predict(letter_group[i][j])
            print(
                f"Font  : {j}\t"
                f"Class : {output}"
            )


if __name__ == "__main__":
    main()
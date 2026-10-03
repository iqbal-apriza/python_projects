import numpy as np
import argparse

import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def read_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        params = yaml.safe_load(file)

    return params


class LVQ:
    def __init__(self, num_inputs, alpha=0.1, dec=1.0):
        self.__num_inputs = num_inputs
        self.__alpha = alpha
        self.__decrease_alpha = dec


    def weight_init(self, x, y):
        self.__weights = []

        for class_ in np.unique(y):
            weight = x[y == class_][0]
            self.__weights.append(weight)

        self.__weights = np.array(self.__weights)
        self.__num_classes = self.__weights.shape[0]


    def preditct(self, input):
        input_len = len(input)

        if input_len != self.__num_inputs:
            raise ValueError(f"Given input length does not match with number of input specified")

        self.__last_input = input

        min_val = np.inf
        idx = np.inf

        for i in range(self.__num_classes):
            sum = 0
            for j in range(self.__num_inputs):
                sum += (self.__last_input[j] - self.__weights[i][j])**2

            if sum < min_val:
                min_val = sum
                idx = i

        self.__last_class = idx
        return self.__last_class


    def update_weight(self, target):
        if self.__last_input is None:
            raise RuntimeError(f"The last input not found. Please predict the data first")

        if self.__last_class is None:
            raise RuntimeError(f"The last class result not found. Please predict the data first")

        if self.__last_class == target:
            for i in range(self.__num_inputs):
                self.__weights[self.__last_class][i] += \
                    self.__alpha * (self.__last_input[i] - self.__weights[self.__last_class][i])

        else:
            for i in range(self.__num_inputs):
                self.__weights[self.__last_class][i] -= \
                    self.__alpha * (self.__last_input[i] - self.__weights[self.__last_class][i])

        self.__alpha *= self.__decrease_alpha


    def get_weight(self):
        return self.__weights


def main():
    ap = argparse.ArgumentParser(description="Python program for LVQ Algorithm")
    ap.add_argument("--dataset", required=True, type=str, help="Dataset file and config in yaml")
    ap.add_argument("--alpha", default=0.1, type=float, help="Learning rate. The value between 0 - 1")
    ap.add_argument("--max-epoch", type=int, help="Max epoch to train")
    ap.add_argument("--decrease-fct", type=float, default=1.0, help="The decreasing factor of learning rate")

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.dataset
    yaml_data = read_yaml(yaml_dir)

    max_epoch = yaml_data.get("max_epoch", args.max_epoch)
    if max_epoch is None:
        ap.error("Please enter max epoch. You can either add it into yaml or use --max-epoch")

    data_in = np.array(yaml_data["input"])
    class_ = np.array(yaml_data["class"])

    alpha = yaml_data.get("alpha", args.alpha)
    decrease_factor = yaml_data.get("decrease_factor", args.decrease_fct)

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    lvq = LVQ(input_col, alpha, decrease_factor)
    lvq.weight_init(data_in, class_)

    for epoch in range(100):
        print(f"--- Executing epoch {epoch+1} ---")
        
        for i in range(input_row):
            lvq.preditct(data_in[i])
            lvq.update_weight(class_[i])

    print(f"\n\n======= RESULT =======\n")
    print(f"{'Input':>15}", end="")
    print(f"{'Pred':>6}", end="")
    print(f"{'Act':>6}", end="\n")

    for i in range(input_row):
        res = lvq.preditct(data_in[i])
        
        print(f"{np.array2string(data_in[i], precision=1):>15}", end="")
        print(f"{res:>6.1f}", end="")
        print(f"{class_[i]:>6.1f}", end="\n")

    print(
        f"\n--- Weights: ---\n"
        f"{lvq.get_weight()}"
    )


if __name__ == '__main__':
    main()
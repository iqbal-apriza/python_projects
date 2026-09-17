import numpy as np
import argparse

import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def read_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        params = yaml.safe_load(file)

    return params


class Kohonen:
    def __init__(self, num_classes, num_input, alpha, decrease_factor):
        self.__num_classes = num_classes
        self.__num_input = num_input

        self.__weights = np.random.uniform(low=0.0, high=1.0, size=(num_classes, num_input))
        self.__alpha = alpha
        self.__decrease_factor = decrease_factor


    def predict(self, input):
        input_len = input.shape[0]

        if input_len != self.__num_input:
            raise ValueError(f"The length of input does not match")

        self.__last_input = input

        min_val = np.inf
        idx = np.inf

        for i in range(self.__num_classes):
            sum = 0
            for j in range(self.__num_input):
                sum += (input[j] - self.__weights[i][j])**2

            if sum < min_val:
                min_val = sum
                idx = i

        self.__last_winner = idx
        return self.__last_winner


    def update(self):
        if self.__last_winner is None:
            raise RuntimeError(f"The last winner is empty. Please train the data first")

        if self.__last_input is None:
            raise RuntimeError(f"The last input is empty. Please train the data first")

        for i in range(self.__num_input):
            self.__weights[self.__last_winner][i] += \
                self.__alpha * \
                (self.__last_input[i] - self.__weights[self.__last_winner][i])

        self.__alpha *= self.__decrease_factor


    def set_weights(self, weights):
        weight_row = weights.shape[0]
        if weight_row != self.__num_classes:
            raise ValueError(f"The row of weights must be the same as number of classes")

        weight_col = weights.shape[1]
        if weight_col != self.__num_input:
            raise ValueError(f"The column of weights must be the same as number of inputs")

        self.__weights = weights


    def get_info(self):
        print(
            f"===== Kohonen Network ====\n"
            f"Number of clases : {self.__num_classes}\n"
            f"Number of inputs : {self.__num_input}\n"
            f"\nWeights:\n"
            f"{self.__weights}"
        )


def main():
    ap = argparse.ArgumentParser(description="Python program for Kohonen Algorithm")
    ap.add_argument("--dataset", required=True, type=str, help="Dataset file that contains input and config in yaml")
    ap.add_argument("--num-classes", type=int, help="Number of classes to be classified")
    ap.add_argument("--alpha", type=float, default=0.1, help="Learning rate. Range between 0 - 1")
    ap.add_argument("--decrease-fct", type=float, default=0.05, help="The decreasing factor of learning rate")
    ap.add_argument("--max-epoch", type=int, help="Max epoch to train")

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.dataset
    yaml_data = read_yaml(yaml_dir)

    data_in = np.array(yaml_data["input"])
    alpha = yaml_data.get("alpha", args.alpha)
    decrease_factor = yaml_data.get("decrease_factor", args.decrease_fct)

    num_classes = yaml_data.get("num_classes", args.num_classes)
    if num_classes is None:
        ap.error("Please enter the number of classes to classified. You can either add it into yaml or use --num-classes")

    max_epoch = yaml_data.get("max_epoch", args.max_epoch)
    if max_epoch is None:
        ap.error("Please enter max epoch. You can either add it into yaml or use --max-epoch")

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    net = Kohonen(num_classes, input_col, alpha, decrease_factor)

    if "weight" in yaml_data:
        weight = np.array(yaml_data["weight"])
        net.set_weights(weight)

    for epoch in range(max_epoch):
        print(f"Executing epoch: {epoch+1}")

        for i in range(input_row):
            net.predict(data_in[i])
            net.update()

    # Predictions
    print("\n\n===== RESULTS =====")
    for i in range(input_row):
        output = net.predict(data_in[i])
        print(
            f"Input : {data_in[i]}\n"
            f"Class : {output}\n"
        )


if __name__ == "__main__":
    main()
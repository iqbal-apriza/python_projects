import numpy as np
import argparse

import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def bipolar_activate(val):
    return 1 if val >= 0 else -1


def evaluate(error):
    err_sqrt = error**2
    mse = np.sum(err_sqrt) / len(error)

    return mse


def read_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        params = yaml.safe_load(file)

    return params


def show_data(input, target, output):
    input_row = input.shape[0]
    input_col = input.shape[1]

    target_row = target.shape[0]
    output_row = output.shape[0]

    if input_row != target_row or input_row != output_row:
        print("Row is not the same")
        return

    for i in range(input_col):
        print(f"{f'Input {i+1}':>8}", end="")

    print(f"{'Target':>8}", end="")
    print(f"{'Output':>8}", end="\n")

    for i in range(input_row):
        for j in range(input_col):
            input_round = round(input[i][j], 3)
            print(f"{input_round:>8}", end="")

        target_round = round(target[i], 3)
        output_round = round(output[i], 3)

        print(f"{target_round:>8}"
              f"{output_round:>8}")

    print("")


class Adaline():
    def __init__(self, num_input):
        self.num_input = num_input
        self.weight = np.random.uniform(low=-1.0, high=1.0, size=self.num_input)
        self.bias = np.random.uniform(low=-1.0, high=1.0)


    def set_weight(self, weight):
        weight_len = weight.shape[0]

        if weight_len != self.num_input:
            raise Exception("The length of given weight does not match\n"
                            f"Given weight length\t: {weight_len}\n"
                            f"Expected length\t\t: {self.num_input}")

        self.weight = weight


    def set_bias(self, bias):
        self.bias = bias


    def train_data(self, input, activation="bipolar"):
        self.recent_input = input
        input_len = self.recent_input.shape[0]

        if input_len != self.num_input:
            raise Exception("The length of given input does not match\n"
                            f"Given input length\t: {input_len}\n"
                            f"Expected length\t\t: {self.num_input}")

        sum = 0
        for i in range(self.num_input):
            sum += self.recent_input[i] * self.weight[i]

        self.__recent_output = sum + self.bias
        if activation == "bipolar":
            net = bipolar_activate(self.__recent_output)

        return net


    def update_weight(self, target, alpha):
        if self.__recent_output is None:
            raise RuntimeError(f"Recent output is empty. Please train the data first")

        if self.recent_input is None:
            raise RuntimeError(f"Recent input is empty. Please train the data first")

        d_w = np.zeros(self.num_input)
        for i in range(self.num_input):
            new_weight = self.weight[i] + alpha * (target - self.__recent_output) * self.recent_input[i]
            d_w[i] = new_weight - self.weight[i]
            self.weight[i] = new_weight

        self.bias = self.bias + alpha * (target - self.__recent_output)

        return d_w


def main():
    ap = argparse.ArgumentParser(description="Python program for Adaline algorithm")
    ap.add_argument("--dataset", required=True, type=str, help="Dataset file and config in yaml")
    ap.add_argument("--alpha", default=0.1, type=float, help="Learning rate. The value between 0 - 1")
    ap.add_argument("--min-err", default=0.1, type=float, help="Minimum error to stop the train")

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.dataset
    yaml_data = read_yaml(yaml_dir)

    data_in = np.array(yaml_data["input"])
    target = np.array(yaml_data["target"])

    alpha = yaml_data.get("alpha", args.alpha)
    min_error = yaml_data.get("min_error", args.min_err)

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    adaline = Adaline(input_col)
    epoch = 0

    while True:
        epoch += 1
        print(f"\n\n--- Epoch ke {epoch} ---")

        error = np.zeros(input_row)
        for i in range(input_row):
            output = adaline.train_data(data_in[i])
            error[i] = (target[i] - output)

            adaline.update_weight(target[i], alpha)

        mse = evaluate(error)
        print(f"MSE : {mse}")

        if mse < min_error:
            break

    print("\n\n===== FINAL RESULT =====")
    output = np.zeros(input_row)
    for i in range(input_row):
        output[i] = adaline.train_data(data_in[i])

    show_data(data_in, target, output)
    print(f"Weights\t: {adaline.weight}")
    print(f"Bias\t: {adaline.bias}")


if __name__ == '__main__':
    main()
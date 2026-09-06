import numpy as np
import argparse
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# data_in = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])

# target = np.array([0, 1, 1, 1])


def read_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        params = yaml.safe_load(file)

    return params


def activation(val, theta):
    return 1 if val >= theta else 0


def net_output(input, weight, theta):
    input_len = input.shape[0]
    weight_len = weight.shape[0]

    if input_len != weight_len:
        print("Length of input is not same as length of weights")
        return None

    sum = 0
    for i in range(input_len):
        sum += input[i] * weight[i]

    return activation(sum, theta)


def update_weight(w_now, alpha, error, input):
    d_w = alpha * error * input
    return w_now + d_w


def evaluate(error):
    err_sqrt = error**2
    mse = np.sum(err_sqrt) / len(error)

    return mse


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
            print(f"{input[i][j]:>8}", end="")

        print(f"{target[i]:>8}"
              f"{output[i]:>8}")

    print("")


def main():
    ap = argparse.ArgumentParser(description="Python program for Delta Rule Algorithm in ANN subject")
    ap.add_argument("--dataset", required=True, type=str, help="Dataset file and config in yaml")
    ap.add_argument("--alpha", default=0.1, type=float, help="Learning rate. The value between 0 - 1")
    ap.add_argument("--theta", default=0.1, type=float, help="Threashold for net activation")

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.dataset
    yaml_data = read_yaml(yaml_dir)

    data_in = np.array(yaml_data["input"])
    target = np.array(yaml_data["target"])
    weights = np.array(yaml_data["weights"])

    alpha = yaml_data.get("alpha", args.alpha)
    theta = yaml_data.get("theta", args.theta)

    data_row = data_in.shape[0]
    data_col = data_in.shape[1]

    output = np.zeros(data_row)
    error = np.zeros(data_row)

    epochs = 0

    while True:
        epochs += 1
        print(f"---- Executing epoch to {epochs} ----")

        for i in range(data_row):
            output[i] = net_output(data_in[i], weights, theta)
            error[i] = target[i] - output[i]

            if error[i] <= 0.01:
                continue

            for j in range(data_col):
                weights[j] = update_weight(weights[j], alpha, error[i], data_in[i][j])

        show_data(data_in, target, output)

        mse = evaluate(error)
        if mse <= 0.1:
            break

    print("\n==== FINAL RESULT ====")
    print(f"Weights\t: {weights}")

    show_data(data_in, target, output)
        

if __name__ == '__main__':
    main()
import numpy as np
import argparse
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def read_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        params = yaml.safe_load(file)

    return params


def activation(val, theta):
    if val > theta:
        return 1
    elif val < -theta:
        return -1
    else:
        return 0


def update_weight(weight, alpha, input, target):
    d_w = alpha * input * target
    return weight + d_w


def update_bias(bias, alpha, target):
    d_b = alpha * target
    return bias + d_b


def calculate_net(input, weight, bias, theta):
    data_len = input.shape[0]
    weight_len = weight.shape[0]

    if data_len != weight_len:
        print("Lenth of data is not same as length of weights")
        return None

    sum = 0
    for i in range(data_len):
        sum += input[i] * weight[i]

    net = sum + bias
    return activation(net, theta)


def evaluate(output, target):
    output_len = output.shape[0]
    target_len = target.shape[0]

    if output_len != target_len:
        print("Length of output is not same as length of target")
        return None

    err_sqrt = (output - target)**2
    mse = np.sum(err_sqrt) / output_len

    if mse <= 0.1:
        return True
    else:
        return False


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
    ap = argparse.ArgumentParser(description="Python program for Perceptron Algorithm in ANN subject")
    ap.add_argument("--dataset", required=True, type=str, help="Dataset file and config in yaml")
    ap.add_argument("--alpha", default=0.1, type=float, help="Learning rate. The value between 0 - 1")
    ap.add_argument("--theta", default=0.1, type=float, help="Threashold for net activation")

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.dataset
    yaml_data = read_yaml(yaml_dir)

    data_in = np.array(yaml_data["input"])
    target = np.array(yaml_data["target"])

    alpha = yaml_data.get("alpha", args.alpha)
    theta = yaml_data.get("theta", args.theta)

    data_row = data_in.shape[0]
    data_col = data_in.shape[1]

    weights = np.zeros(data_col)
    bias = 0

    epochs = 0
    result = None

    while True:
        epochs += 1
        print(f"---- Executing epoch to {epochs} ----")

        net_out = np.zeros(data_row)

        for i in range(data_row):
            net_out[i] = calculate_net(data_in[i], weights, bias, theta)

            if net_out[i] == target[i]:
                continue

            bias = update_bias(bias, alpha, target[i])
            for j in range(data_col):
                weights[j] = update_weight(weights[j], alpha, data_in[i][j], target[i])

        show_data(data_in, target, net_out)
        is_finish = evaluate(net_out, target)
        if is_finish:
            result = net_out
            break

    print("\n==== FINAL RESULT ====")
    print(f"Weights\t: {weights}\n"
          f"Bias\t: {bias}")

    show_data(data_in, target, result)


if __name__ == '__main__':
    main()
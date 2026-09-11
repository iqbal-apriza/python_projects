import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import yaml
from pathlib import Path

import sys

if sys.platform == "win32":
    import msvcrt
else:
    import select
    import termios
    import tty

from backpropagation import backpropagation as nn

BASE_DIR = Path(__file__).resolve().parent


def read_dataset(csv_file):
    df = pd.read_csv(csv_file)
    df_norm = df.drop(df.columns[0], axis=1)

    return df_norm.to_numpy()


def normalize_data(data):
    max_data = np.max(data)
    min_data = np.min(data)

    return 0.8 * (data - min_data) / (max_data - min_data) + 0.1


def main():
    ap = argparse.ArgumentParser(description="Python program for rainfall forecasting using Backpropagation ANN")
    ap.add_argument("--dataset", required=True, type=str, help="Dataset file and config in yaml")
    ap.add_argument("--export-dir", type=str, help="Directory for exporting the weights and biases")
    ap.add_argument("--alpha", default=0.1, type=float, help="Learning rate. The value between 0 - 1")
    ap.add_argument("--mu", default=0.0, type=float, help="Momentum coefficient for updating weights")
    ap.add_argument("--max-epoch", type=int, help="Maximum epoch to stop train")
    ap.add_argument("--min-err", default=0.1, type=float, help="Minimum error to stop the train")

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.dataset
    yaml_data = nn.read_yaml(yaml_dir)

    dataset_dir = BASE_DIR / yaml_data["dataset"]
    dataset = read_dataset(dataset_dir)

    dataset_norm = normalize_data(dataset)
    data_col = dataset_norm.shape[1]

    data_in = dataset_norm[:, :data_col-1]
    target = dataset_norm[:, data_col-1:].ravel()

    num_layer = yaml_data["num_layer"]
    num_neurons = np.array(yaml_data["num_neurons"])

    alpha = yaml_data.get("alpha", args.alpha)
    mu = yaml_data.get("mu", args.mu)
    min_error = yaml_data.get("min_error", args.min_err)
    max_epoch = yaml_data.get("max_epoch", args.max_epoch)

    if args.export_dir is not None:
        export_dir = BASE_DIR / args.export_dir

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    layers = [[] for _ in range(num_layer)]
    for i in range(num_layer):
        if i == 0:
            layers[i] = nn.Layer(num_neurons[i], input_col, i)
            layers[i].nguyen_widrow_init()

        elif i == num_layer - 1:
            layers[i] = nn.Layer(1, num_neurons[i-1], i)

        else:
            layers[i] = nn.Layer(num_neurons[i], num_neurons[i-1], i)

    epochs = 0
    mse_acc = []
    epochs_acc = []

    if sys.platform != "win32":
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    print("\nTraining the data. Press <q> to stop the training process")

    stop_train = False

    try:
        while True:
            key = nn.check_keyboard()

            if key is not None and key.lower() == 'q':
                stop_train = True

            epochs += 1

            error = np.zeros(input_row)
            output = np.zeros(input_row)

            for i in range(input_row):
                net_out = [[] for _ in range(num_layer)]
                for j in range(num_layer):
                    if j == 0:
                        net_out[j] = layers[j].forward_prop(data_in[i])

                    else:
                        net_out[j] = layers[j].forward_prop(net_out[j-1])

                error[i] = target[i] - net_out[num_layer - 1][0]

                backprop_err = [[] for _ in range(num_layer)]
                for j in range(num_layer):
                    rev_count = num_layer - j

                    if j == 0:
                        backprop_err[j] = layers[rev_count - 1].back_prop(np.array([error[i]]), alpha, net_out[rev_count - 2], mu)

                    elif j == num_layer - 1:
                        backprop_err[j] = layers[rev_count - 1].back_prop(backprop_err[j-1], alpha, data_in[i], mu)

                    else:
                        backprop_err[j] = layers[rev_count - 1].back_prop(backprop_err[j-1], alpha, net_out[rev_count - 2], mu)
                        
            mse = nn.evaluate(error)
            print(f"\033[2K\rEpoch to {epochs}\n"
                    f"\033[2K\rMSE\t: {mse:.7f}\n"
                    f"\033[2K\rTarget\t: {min_error}")

            if nn.is_save(epochs):
                epochs_acc.append(epochs)
                mse_acc.append(mse)

            # if mse <= min_error:
            #     stop_train = True

            if max_epoch and epochs >= max_epoch:
                stop_train = True

            if stop_train:
                break
            else:
                if nn.is_save(epochs):
                    print("")
                else:
                    print("\033[3A", end="")

    finally:
        if sys.platform != "win32":
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    print("\n\n====================================\n"
            "            FINAL RESULT            \n"
            "====================================\n")
    
    for i in range(num_layer):
        layers[i].get_info()

        if args.export_dir is not None:
            layers[i].export_weights(export_dir)

    output = np.zeros(input_row)
    for i in range(input_row):
        net_out = [[] for _ in range(num_layer)]
        for j in range(num_layer):
            if j == 0:
                net_out[j] = layers[j].forward_prop(data_in[i])

            elif j == num_layer - 1:
                output[i] = layers[j].forward_prop(net_out[j-1])

            else:
                net_out[j] = layers[j].forward_prop(net_out[j-1])

    nn.show_data(data_in, target, output)

    plt.plot(epochs_acc, mse_acc)
    plt.title("MSE vs Epoch Plot")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")

    plt.show()
    


if __name__ == '__main__':
    main()
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

from backpropagation import backpropagation as bp

BASE_DIR = Path(__file__).resolve().parent


class Dataset():
    def __init__(self, dataset):
        self.df = pd.read_csv(dataset)
        self.df_np = self.df.to_numpy()

        self.min = np.min(self.df_np)
        self.max = np.max(self.df_np)

        self.data_row = self.df_np.shape[0]
        self.data_col = self.df_np.shape[1]


    def get_data(self):
        data_in = self.df_np[:, :self.data_col-1]
        target = self.df_np[:, self.data_col-1:].ravel()

        return data_in, target


    def get_norm_data(self):
        data_norm = 0.8 * (self.df_np - self.min) / (self.max - self.min) + 0.1

        data_in = data_norm[:, :self.data_col-1]
        target = data_norm[:, self.data_col-1:].ravel()

        return data_in, target


def main():
    ap = argparse.ArgumentParser(description='Final project forcasting using Backpropagation')
    ap.add_argument('--dataset', type=str, required=True, help='Dataset file in csv')
    ap.add_argument('--config', type=str, required=True, help='Config file for ann architecture in yaml')
    ap.add_argument('--export-dir', type=str, help='Directory for exporting the weights and biases')

    args = ap.parse_args()

    dataset_dir = BASE_DIR / args.dataset
    data = Dataset(dataset_dir)

    data_in, target = data.get_norm_data()

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    yaml_dir = BASE_DIR / args.config
    yaml_data = bp.read_yaml(yaml_dir)

    alpha = yaml_data.get('alpha', 0.1)
    mu = yaml_data.get('mu', 0.0)
    min_error = yaml_data.get('min_error', False)
    max_epoch = yaml_data.get('max_epoch', False)

    num_layer = yaml_data['num_layer']
    num_neurons = np.array(yaml_data['num_neurons'])

    if args.export_dir is not None:
        export_dir = BASE_DIR / args.export_dir

    layers = [[] for _ in range(num_layer)]
    for i in range(num_layer):
        if i == 0:
            layers[i] = bp.Layer(num_neurons[i], input_col, i)
            if yaml_data.get('nguyen_widrow', False):
                layers[i].nguyen_widrow_init()

        elif i == num_layer - 1:
            layers[i] = bp.Layer(1, num_neurons[i-1], i)

        else:
            layers[i] = bp.Layer(num_neurons[i], num_neurons[i-1], i)

    epochs = 0
    mse_acc = []
    epochs_acc = []
    last_mse = np.inf

    if sys.platform != "win32":
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    print("\nTraining the data. Press <q> to stop the training process")

    stop_train = False

    try:
        while True:
            key = bp.check_keyboard()

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

            mse = bp.evaluate(error)
            print(f"\033[2K\rEpoch to {epochs}\n"
                  f"\033[2K\rMSE\t: {mse:.7f}\n"
                  f"\033[2K\rTarget\t: {min_error}")

            if bp.is_save(epochs):
                epochs_acc.append(epochs)
                mse_acc.append(mse)

            if max_epoch and epochs >= max_epoch:
                stop_train = True

            if min_error and mse <= min_error:
                stop_train = True

            if stop_train:
                last_mse = mse
                break
            else:
                if bp.is_save(epochs):
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

    bp.show_data(data_in, target, output)

    plt.plot(epochs_acc, mse_acc)
    plt.title("MSE vs Epoch Plot")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")

    plt.show()

    msg_buffer = (
        f"{'='*50}\n"
        f"{' '*17}TRAINING INFO\n"
        f"{'='*50}\n"

        f"Network Architecture\n\n"
        f"Number of Layers{' '*8}: {num_layer}\n"
        f"Number of Neurons{' '*7}: {num_neurons}\n"

        f"{'-'*50}\n"

        f"Hyperparameters\n\n"
        f"Learning Rate{' '*11}: {alpha}\n"
        f"Momentum{' '*16}: {mu}\n"
        f"Nguyen-Widrow{' '*11}: {yaml_data.get('nguyen_widrow', False)}\n"
        f"Max Epoch{' '*15}: {max_epoch}\n"
        f"Min Error{' '*15}: {min_error}\n"

        f"{'-'*50}\n"

        f"Training Info\n\n"
        f"Stop Epochs{' '*13}: {epochs}\n"
        f"Mean Squared Errors{' '*5}: {last_mse}\n"

        f"{'-'*50}\n"
    )

    print("\n\n")
    print(msg_buffer)

    if args.export_dir is not None:
        file_dir = export_dir / "log.txt"
        
        with open(file_dir, "w", encoding="utf-8") as f:
            f.write(msg_buffer)


if __name__ == '__main__':
    main()
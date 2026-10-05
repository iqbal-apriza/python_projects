import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import yaml
from pathlib import Path

from backpropagation import backpropagation as bp

BASSE_DIR = Path(__file__).resolve().parent


def get_mape(prediction, actual):
    if len(prediction) != len(actual):
        raise RuntimeError(f"The length of data does not match")

    sum = 0
    for i in range(len(prediction)):
        sum += np.abs((actual[i] - prediction[i]) / actual[i])

    return 100 * sum / len(prediction)


def get_r2(prediction, actual):
    if len(prediction) != len(actual):
        raise RuntimeError(f"The length of data does not match")
    
    ss_res = 0
    ss_tot = 0
    act_avg = np.average(actual)

    for i in range(len(prediction)):
        ss_res += (actual[i] - prediction[i])**2
        ss_tot += (actual[i] - act_avg)**2

    return 1 - (ss_res / ss_tot)


class Dataset():
    def __init__(self, dataset):
        self.df = pd.read_csv(dataset)
        self.df_np = self.df.to_numpy()

        self.min = np.min(self.df_np)
        self.max = np.max(self.df_np)

        self.data_row = self.df_np.shape[0]
        self.data_col = self.df_np.shape[1]


    def split_data(self, split_col, norm=False):
        data = self.df_np
        if norm:
            data = 0.8 * (self.df_np - self.min) / (self.max - self.min) + 0.1

        left = data[:, :split_col]
        right = data[:, split_col:]

        return left, right


    def append_column(self, col_name, data):
        self.df[col_name] = data


def main():
    ap = argparse.ArgumentParser(description='Program to calculate model accuracy from previous train')
    ap.add_argument('--model', type=str, required=True, help='Model directory from previous train')
    ap.add_argument('--config', type=str, required=True, help='Architecture config file directory in yaml')
    ap.add_argument('--dataset', type=str, required=True, help='Dataset file directory in csv')
    ap.add_argument('--save-plt', action='store_true', help='Save the plot result into model directory')
    ap.add_argument('--plt-name', type=str, help='The plot name to displaying')

    args = ap.parse_args()

    model_dir = BASSE_DIR / args.model
    config_dir = BASSE_DIR / args.config
    dataset_dir = BASSE_DIR / args.dataset

    config_data = bp.read_yaml(config_dir)
    num_layers = config_data["num_layer"]
    num_neurons = np.array(config_data["num_neurons"])

    data = Dataset(dataset_dir)
    data_in, target = data.split_data(split_col=8, norm=True)
    target = target.ravel()

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    layers = [[] for _ in range(num_layers)]
    for i in range(num_layers):
        if i == 0:
            layers[i] = bp.Layer(num_neurons[i], input_col, i)

        elif i == num_layers - 1:
            layers[i] = bp.Layer(1, num_neurons[i-1], i)

        else:
            layers[i] = bp.Layer(num_neurons[i], num_neurons[i-1], i)

        weight_dir = model_dir / f"layer_{i}" / "weights.npy"
        bias_dir = model_dir / f"layer_{i}" / "biases.npy"

        weights = np.load(weight_dir)
        biases = np.load(bias_dir)

        layers[i].set_weights(weights)
        layers[i].set_bias(biases)

    output = np.zeros(input_row)
    for i in range(input_row):
        net_out = [[] for _ in range(num_layers)]
        for j in range(num_layers):
            if j == 0:
                net_out[j] = layers[j].forward_prop(data_in[i])

            else:
                net_out[j] = layers[j].forward_prop(net_out[j-1])

        output[i] = net_out[num_layers - 1]

    accuracy = 100 - get_mape(output, target)
    r2 = get_r2(output, target)

    fig, ax = plt.subplots()

    ax.plot(target, marker='o', label='Actual', alpha=0.7, linewidth=2)
    ax.plot(output, marker='o', label='Prediction', alpha=0.7)

    ax.grid(True)

    plt_name = ' '
    if args.plt_name is not None:
        plt_name = args.plt_name + ' '

    ax.set_title(f"{plt_name}Prediction Accuracy of 2005")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{plt_name}Norm")

    ax.set_xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )

    ax.legend()
    ax.text(
        0.05, 0.95,
        f"Accuracy : {accuracy:.2f}%\n"
        f"R2 : {r2:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    if args.save_plt is not None:
        file_dir = model_dir / 'prediction.png'
        plt.savefig(file_dir)

    plt.show()


if __name__ == '__main__':
    main()
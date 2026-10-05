import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import yaml
from pathlib import Path

from backpropagation import backpropagation as bp

BASE_DIR = Path(__file__).resolve().parent


def denormalize_data(data, data_min, data_max):
    return (data_max * (data - 0.1) + data_min * (0.9 - data)) / 0.8


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
    ap = argparse.ArgumentParser(description='Forcasting the data')
    ap.add_argument('--model', type=str, required=True, help='Model directory from previous train')
    ap.add_argument('--config', type=str, required=True, help='Architecture config file directory in yaml')
    ap.add_argument('--dataset', type=str, required=True, help='Dataset file directory in csv')

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.config
    yaml_data = bp.read_yaml(yaml_dir)

    dataset_dir = BASE_DIR / args.dataset
    data = Dataset(dataset_dir)
    _, data_in = data.split_data(split_col=1, norm=True)

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]    

    train_dir = BASE_DIR / args.model

    num_layers = yaml_data["num_layer"]
    num_neurons = np.array(yaml_data["num_neurons"])

    layers = [[] for _ in range(num_layers)]
    for i in range(num_layers):
        if i == 0:
            layers[i] = bp.Layer(num_neurons[i], input_col, i)

        elif i == num_layers - 1:
            layers[i] = bp.Layer(1, num_neurons[i-1], i)

        else:
            layers[i] = bp.Layer(num_neurons[i], num_neurons[i-1], i)

        weight_dir = train_dir / f"layer_{i}" / "weights.npy"
        bias_dir = train_dir / f"layer_{i}" / "biases.npy"

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

    data_denorm = denormalize_data(output, data.min, data.max)
    data_denorm = np.round(data_denorm, 1)

    data.append_column(col_name='2006', data=data_denorm)

    print(data.df)
    

if __name__ == '__main__':
    main()
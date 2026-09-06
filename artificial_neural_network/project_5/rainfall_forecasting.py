import numpy as np
import pandas as pd

import argparse
import yaml
from pathlib import Path

from project_4 import backpropagation as nn
from project_5 import rainfall_training as tr

BASE_DIR = Path(__file__).resolve().parent


def denormalize_data(data, data_min, data_max):
    return (data_max * (data - 0.1) + data_min * (0.9 - data)) / 0.8


def main():
    ap = argparse.ArgumentParser(description="Python program for forcasting the rainfall using Backpropagation ANN")
    ap.add_argument("--train-dir", default="train", type=str, help="Training directory that contains weights and biases from training")
    ap.add_argument("--config", default="config.yaml", type=str, help="Configuration file of ANN architecture in yaml")

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.config
    yaml_data = nn.read_yaml(yaml_dir)

    dataset_dir = BASE_DIR / yaml_data["dataset"]
    dataset = tr.read_dataset(dataset_dir)
    dataset_norm = tr.normalize_data(dataset)

    data_col = dataset_norm.shape[1]
    data_in = dataset_norm[:, 1:]

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    num_layers = yaml_data["num_layer"]
    num_neurons = np.array(yaml_data["num_neurons"])

    train_dir = BASE_DIR / args.train_dir

    layers = [[] for _ in range(num_layers)]
    for i in range(num_layers):
        if i == 0:
            layers[i] = nn.Layer(num_neurons[i], input_col, i)

        elif i == num_layers - 1:
            layers[i] = nn.Layer(1, num_neurons[i-1], i)

        else:
            layers[i] = nn.Layer(num_neurons[i], num_neurons[i-1], i)

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

    data_denorm = denormalize_data(output, np.min(dataset), np.max(dataset))

    df = pd.read_csv(dataset_dir)
    df["2006"] = np.round(data_denorm, 1)

    print(df)


if __name__ == '__main__':
    main()
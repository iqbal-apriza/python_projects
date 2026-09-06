import numpy as np
import matplotlib.pyplot as plt

import argparse
import yaml
from pathlib import Path

import sys
import select
import termios
import tty

BASE_DIR = Path(__file__).resolve().parent


def read_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        params = yaml.safe_load(file)

    return params


def activation(val):
    return 1 / (1 + np.exp(-val))


def evaluate(error):
    err_sqrt = error**2
    mse = np.sum(err_sqrt) / len(error)

    return mse


def is_save(epoch):
    if epoch < 10:
        return True

    interval = 10 ** (len(str(epoch)) - 1)
    return epoch % interval == 0


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

        output_round = round(output[i], 3)

        print(f"{target[i]:>8}"
              f"{output_round:>8}")

    print("")


class Neuron:
    def __init__(self, num_input):
        self.__num_input = num_input
        self.__weights = np.random.uniform(low=-0.5, high=0.5, size=num_input)
        self.__bias = np.random.uniform(low=-0.5, high=0.5)

        self.__prev_weights = np.zeros(num_input)
        self.__prev_bias = 0.0


    def set_init_weight(self, val):
        if len(val) != self.__num_input:
            print("The given weight length is not the same as number of input. Operation discarded")

        self.__weights = val


    def set_init_bias(self, val):
        self.__bias = val


    def get_weights(self):
        return self.__weights


    def get_bias(self):
        return self.__bias


    def get_net_output(self, input):
        input_len = len(input)

        if input_len != self.__num_input:
            print("The given input length is not the same as number of input. Operation discarded")

        sum = 0
        for i in range(input_len):
            sum += input[i] * self.__weights[i]

        sum += self.__bias
        self.recent_net = activation(sum)

        return self.recent_net


    def get_derror(self, error):
        self.recent_derr = error * self.recent_net * (1 - self.recent_net)
        return self.recent_derr


    def update_weight(self, alpha, input, mu=0.0):
        input_len = len(input)

        if input_len != self.__num_input:
            print("The given input length is not the same as number of input. Operation discarded")

        for i in range(self.__num_input):
            d_w = alpha * self.recent_derr * input[i]

            d_w += mu * (self.__weights[i] - self.__prev_weights[i])
            self.__prev_weights[i] = self.__weights[i]

            self.__weights[i] += d_w

    def update_bias(self, alpha, mu=0.0):
        d_b = alpha * self.recent_derr
        self.__bias += d_b


class Layer:
    def __init__(self, num_neuron, num_input, id):
        self.__num_neuron = num_neuron
        self.__num_input = num_input
        self.__id = id

        self.__neurons = [Neuron(num_input) for _ in range(num_neuron)]


    def forward_prop(self, input):
        input_len = len(input)

        if input_len != self.__num_input:
            print(f"[Layer:{self.__id}] The given input length is not the same as number of input. Operation discarded")

        net_output = np.zeros(self.__num_neuron)
        for i in range(self.__num_neuron):
            net_output[i] = self.__neurons[i].get_net_output(input)

        return net_output


    def back_prop(self, error, alpha, input, mu=0.0):
        d_err = np.zeros(self.__num_neuron)
        for i in range(self.__num_neuron):
            d_err[i] = self.__neurons[i].get_derror(error[i])

        neuron_weights = [[] for _ in range(self.__num_neuron)]
        for i in range(self.__num_neuron):
            neuron_weights[i] = self.__neurons[i].get_weights()

        d_net = np.zeros(self.__num_input)
        for i in range(self.__num_input):
            sum = 0
            for j in range(self.__num_neuron):
                sum += d_err[j] * neuron_weights[j][i]

            d_net[i] = sum

        for i in range(self.__num_neuron):
            self.__neurons[i].update_weight(alpha, input, mu)
            self.__neurons[i].update_bias(alpha)

        return d_net


    def get_weights(self):
        neuron_weights = [[] for _ in range(self.__num_neuron)]
        neuron_biases = np.zeros(self.__num_neuron)
        for i in range(self.__num_neuron):
            neuron_weights[i] = self.__neurons[i].get_weights()
            neuron_biases[i] = self.__neurons[i].get_bias()

        weight_and_bias = np.array(neuron_weights).T
        weight_and_bias = np.vstack((weight_and_bias, neuron_biases))

        return weight_and_bias


    def get_info(self):
        print(f"\n===== LAYER to {self.__id} =====\n"
              f"Num Neurons\t: {self.__num_neuron}\n"
              f"Num Inputs\t: {self.__num_input}\n")

        for i in range(self.__num_neuron):
            weight = self.__neurons[i].get_weights()
            bias = self.__neurons[i].get_bias()
            print(f"--- Neuron {i} ---\n"
                  f"Weight\t: {weight}\n"
                  f"Bias\t: {bias}\n")

        print()


def main():
    ap = argparse.ArgumentParser(description="Python program for Backpropagation Algorithm in ANN subject")
    ap.add_argument("--dataset", required=True, type=str, help="Dataset file and config in yaml")
    ap.add_argument("--alpha", default=0.1, type=float, help="Learning rate. The value between 0 - 1")
    ap.add_argument("--mu", default=0.0, type=float, help="Momentum coefficient for updating weights")
    ap.add_argument("--max-epoch", type=int, help="Maximum epoch to stop train")
    ap.add_argument("--min-err", default=0.1, type=float, help="Minimum error to stop the train")

    args = ap.parse_args()

    yaml_dir = BASE_DIR / args.dataset
    yaml_data = read_yaml(yaml_dir)

    data_in = np.array(yaml_data["input"])
    target = np.array(yaml_data["target"])
    num_layer = yaml_data["num_layer"]
    num_neurons = np.array(yaml_data["num_neurons"])

    if num_layer != len(num_neurons):
        raise ValueError("Number of neurons in array must be the same as number of layers")

    alpha = yaml_data.get("alpha", args.alpha)
    mu = yaml_data.get("mu", args.mu)
    min_error = yaml_data.get("min_error", args.min_err)
    max_epoch = yaml_data.get("max_epoch", args.max_epoch)

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    layers = [[] for _ in range(num_layer)]
    for i in range(num_layer):
        if i == 0:
            layers[i] = Layer(num_neurons[i], input_col, i)

        elif i == num_layer - 1:
            layers[i] = Layer(1, num_neurons[i-1], i)

        else:
            layers[i] = Layer(num_neurons[i], num_neurons[i-1], i)

    epochs = 0
    mse_acc = []
    epochs_acc = []

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    print("\nTraining the data. Press <q> to stop the training process")

    stop_train = False

    try:
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)

                if key.lower() == 'q':
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
                        
            mse = evaluate(error)
            print(f"\033[2K\rEpoch to {epochs}\n"
                  f"\033[2K\rMSE\t: {mse:.7f}\n"
                  f"\033[2K\rTarget\t: {min_error}")

            if is_save(epochs):
                epochs_acc.append(epochs)
                mse_acc.append(mse)

            if mse <= min_error:
                stop_train = True

            if max_epoch and epochs >= max_epoch:
                stop_train = True

            if stop_train:
                break
            else:
                if is_save(epochs):
                    print("")
                else:
                    print("\033[3A", end="")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    print("\n\n====================================\n"
          "            FINAL RESULT            \n"
          "====================================\n")
    for i in range(num_layer):
        layers[i].get_info()

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

    show_data(data_in, target, output)

    plt.plot(epochs_acc, mse_acc)
    plt.title("MSE vs Epoch Plot")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")

    plt.show()


if __name__ == '__main__':
    main()
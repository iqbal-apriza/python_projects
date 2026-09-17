import numpy as np
import argparse

import yaml
from pathlib import Path

import adaline as ad


class Madaline():
    def __init__(self, num_uints, num_inputs, id:int):
        self.__num_units = num_uints
        self.__num_inputs = num_inputs
        self.__id = id

        self.__nets = [ad.Adaline(self.__num_inputs) for _ in range(num_uints)]

        if self.__id > 0:
            weight = np.full(shape=(self.__num_units, self.__num_inputs), fill_value=0.5)
            bias = 0.5

            for i in range(self.__num_units):
                self.__nets[i].set_weight(weight[i])
                self.__nets[i].set_bias(bias)


    def train(self, input):
        input_len = input.shape[0]

        if input_len != self.__num_inputs:
            raise ValueError(f"The given input length is not the same as number of input")

        net_output = np.zeros(self.__num_units)
        for i in range(self.__num_units):
            net_output[i] = self.__nets[i].train_data(input)

        return net_output


    def update_weights(self, target, alpha):
        for i in range(self.__num_units):
            self.__nets[i].update_weight(target, alpha)


    def set_weights(self, weight):
        weight_row = weight.shape[0]
        weight_col = weight.shape[1]

        if weight_row != self.__num_units:
            raise ValueError(f"Weight row must be the same as number of units")

        if weight_col != self.__num_inputs:
            raise ValueError(f"Weight column must be the same as number of inputs")

        for i in range(self.__num_units):
            self.__nets[i].set_weight(weight[i])


    def set_biases(self, bias):
        if bias.shape != self.__num_units:
            raise ValueError(f"Length of bias must be the same as number of units")

        for i in range(self.__num_units):
            self.__nets[i].set_bias(bias[i])


    def get_info(self):
        print(f"===== Madaline layer {self.__id} =====")
        print(f"Number of inputs : {self.__num_inputs}")
        print(f"Number of units\t : {self.__num_units}")

        for i in range(self.__num_units):
            print(f"\n--- Unit {i} ---")
            print(f"Weights\t: {self.__nets[i].weight}")
            print(f"Bias\t: {self.__nets[i].bias}")


def main():
    data_in = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
              [-1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])

    target = np.array([1, 1, 1, -1])

    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    num_layers = 2
    num_neurons = np.array([2, 1])

    hidden_layer = Madaline(num_neurons[0], input_col, 0)
    output_layer = Madaline(num_neurons[1], num_neurons[0], 1)

    # weight_w = np.array([
    #     [0.05, 0.2],
    #     [0.1, 0.2]
    # ])

    # bias_w = np.array([0.3, 0.15])

    # hidden_layer.set_weights(weight_w)
    # hidden_layer.set_biases(bias_w)

    # weight_v = np.array([
    #     [0.5, 0.5]
    # ])

    # bias_v = np.array([0.5])

    # output_layer.set_weights(weight_v)
    # output_layer.set_biases(bias_v)

    epochs = 0

    while True:
        epochs += 1
        print(f"==== Executing Epoch {epochs} ====")

        error = np.zeros(input_row)
        for i in range(input_row):
            output_z = hidden_layer.train(data_in[i])
            output_y = output_layer.train(output_z)

            error[i] = target[i] - output_y
            if error[i] == 0:
                continue

            hidden_layer.update_weights(target[i], 0.0001)

        print(f"Errors: {error}", end="\n\n")
        
        if epochs >= 3:
            break

    print(hidden_layer.get_info())
    print("\n")
    print(output_layer.get_info())
        

if __name__ == "__main__":
    main()
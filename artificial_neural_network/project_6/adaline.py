import numpy as np
import matplotlib.pyplot as plt
import argparse

data_in = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

target = np.array([1, -1, -1, 1])

alpha = 0.1
tolerance = 0.05


def activate(val):
    return 1 if val >= 0 else -1


def update_bias(bias, target, output, alpha):
    return bias + (alpha * (target - output))


def update_weight(weight, target, input, output, alpha):
    return weight + (alpha * (target - output)  * input)


def main():
    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    weight = np.random.uniform(low=0, high=1, size=input_col)
    bias = np.random.uniform(low=0, high=1)

    epochs = 0

    while True:
        epochs += 1
        print(f"Epoch to {epochs}")

        d_w = np.zeros(input_col)

        for i in range(input_row):
            sum = 0
            for j in range(input_col):
                sum += data_in[i][j] * weight[j]

            sum += bias
            output = activate(sum)

            new_bias = update_bias(bias, target[i], output, alpha)
            bias = new_bias
            for j in range(input_col):
                new_weight = update_weight(weight[j], target[i], data_in[i][j], output, alpha)
                d_w[j] = new_weight - weight[j]

                print(d_w[j])

                weight[j] = new_weight

        print("Current Weight")
        print(weight[0])
        print(d_w[0])
        print("")
        # print(weight[1])
        # print("")

        if np.abs(d_w[0]) < tolerance:
            break

if __name__ == '__main__':
    main()
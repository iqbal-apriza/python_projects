import numpy as np

data_in = np.array([
    [-1 ,-1],
    [-1, 1],
    [1, -1],
    [1, 1]
])

target = np.array([-1, -1, -1, 1])


def update_weight(w_now, input, target):
    return w_now + input * target


def update_bias(b_now, target):
    return b_now + target


def activate(val):
    return 1 if val >= 0 else -1


def calculate_net(input, weight, bias):
    data_len = input.shape[0]
    weight_len = weight.shape[0]

    if data_len != weight_len:
        print("Length of data is not same as length of weights")
        return None

    sum = 0
    for i in range(data_len):
        sum += input[i] * weight[i]

    net = sum + bias
    return activate(net)


def main():
    weights = np.array([0, 0])
    bias = 0

    data_row = data_in.shape[0]
    data_col = data_in.shape[1]

    for i in range(data_row):
        print(f"--- Calculating for data {i+1} ---")
        bias = update_bias(bias, target[i])

        for j in range(data_col):
            weights[j] = update_weight(weights[j], data_in[i][j], target[i])

        print(f"Weights\t: {weights}\n"
              f"Bias\t: {bias}\n")

    print("==== Final Result ====")
    print(f"Weights\t: {weights}\n"
          f"Bias\t: {bias}\n")

    print(f"{'Input 1':>8}{'Input 2':>8}{'Target':>8}{'Output':>8}")
    result = np.zeros(data_row)
    for i in range(data_row):
        result[i] = calculate_net(data_in[i], weights, bias)
        print(f"{data_in[i][0]:>8}{data_in[i][1]:>8}{target[i]:>8}{result[i]:>8}")


if __name__ == '__main__':
    main()
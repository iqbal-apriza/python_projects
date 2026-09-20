import numpy as np
from prettytable import PrettyTable

inputt = [
    [1, 1, -1, -1],
    [1, -1, 1, -1]
]

target = [-1, 1, 1, -1]

alpha = 0.5
net_h = np.zeros([2])
output_h = np.zeros([2])
output = [0, 0, 0, 0]

# np.random.seed(42)
weight_w = [
    [-0.06663228, -0.01719245],
    [-0.30715648, 0.2209582],
    [0.44151396, -0.1058249]
]

weight_v = [0.09838382, -0.39990655, 0.14156771]

table1 = PrettyTable(["X1", "X2", "Target", "Out Z1", "Out Z2"])
table2 = PrettyTable(["Z1", "Z2", "Target", "Out", "Error"])

epoch = 0

while output != target:
    epoch = epoch + 1
    print(f"----------------Epoch ke : {epoch}----------------")
    for i in range(4):
        for j in range(2):
            net_h[j] = weight_w[0][j] + (inputt[0][i] * weight_w[1][j]) + (inputt[1][i] * weight_w[2][j])
            if net_h[j] >= 0:
                output_h[j] = 1
            elif net_h[j] < 0:
                output_h[j] = -1

        net = weight_v[0] + (output_h[0] * weight_v[1]) + (output_h[1] * weight_v[2])
        if net >= 0:
            output[i] = 1
        elif net < 0:
            output[i] = -1

        error = target[i] - output[i]

        if error != 0:
            for j in range(2):
                for k in range(2):
                    weight_w[k][j] = weight_w[k][j] + (alpha * (target[i] - output_h[j]) * inputt[k][i])
                weight_w[2][j] = weight_w[2][j] + (alpha * (target[i] - output_h[j]))

                weight_v[j+1] = weight_v[j+1] + (alpha * error * output_h[j])
            weight_v[0] = weight_v[0] + (alpha * error)

        table1.add_row([inputt[0][i], inputt[1][i], target[i], output_h[0], output_h[1]])
        table2.add_row([output_h[0], output_h[1], target[i], output[i], error])

    print(table1)
    print(table2)
    table1.clear_rows()
    table2.clear_rows()
    print("")
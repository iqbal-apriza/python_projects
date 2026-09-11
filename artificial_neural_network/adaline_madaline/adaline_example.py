import numpy as np

data_in = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

target = np.array([1, -1, -1, -1])

alpha = 0.1
threshold = 0.01


class Adaline():
    def __init__(self, num_input):
        self.num_input = num_input
        self.weight = np.random.uniform(low=-1.0, high=1.0, size=self.num_input)
        self.bias = np.random.uniform(low=-1.0, high=1.0)


    def set_weight(self, weight):
        weight_len = weight.shape[0]

        if weight_len != self.num_input:
            raise Exception("The length of given weight does not match\n"
                            f"Given weight length\t: {weight_len}\n"
                            f"Expected length\t\t: {self.num_input}")

        self.weight = weight


    def set_bias(self, bias):
        self.bias = bias


    def train_data(self, input):
        self.recent_input = input
        input_len = self.recent_input.shape[0]

        if input_len != self.num_input:
            raise Exception("The length of given input does not match\n"
                            f"Given input length\t: {input_len}\n"
                            f"Expected length\t\t: {self.num_input}")

        sum = 0
        for i in range(self.num_input):
            sum += self.recent_input[i] * self.weight[i]

        return sum + self.bias


    def update_weight(self, target, output, alpha):
        d_w = np.zeros(self.num_input)
        for i in range(self.num_input):
            new_weight = self.weight[i] + alpha * (target - output) * self.recent_input[i]
            d_w[i] = new_weight - self.weight[i]
            self.weight[i] = new_weight

        self.bias = self.bias + alpha * (target - output)

        return d_w


def main():
    input_row = data_in.shape[0]
    input_col = data_in.shape[1]

    adaline = Adaline(input_col)

    # weight = np.random.uniform(low=0.0, high=1.0, size=2)
    # bias = np.random.uniform(low=0.0, high=1.0)

    epoch = 0
    stop_train = False

    while stop_train is False:
        epoch += 1
        print(f"\n\n--- Epoch ke {epoch} ---")

        for i in range(input_row):
            output = adaline.train_data(data_in[i])
            
            d_w = adaline.update_weight(target[i], output, alpha)
            print(f"Iterasi ke {i+1}")
            print(d_w)

        if epoch >= 10:
            break

        # for i in range(input_row):
        #     sum = 0
        #     for j in range(input_col):
        #         sum += data_in[i][j] * weight[j]

        #     output = sum + bias

        #     bias = bias + alpha * (target[i] - output)
        #     counter = 0
        #     for j in range(input_col):
        #         new_weight = weight[j] + alpha * (target[i] - output) * data_in[i][j]
        #         d_w = new_weight - weight[j]
        #         weight[j] = new_weight

        #         if np.abs(d_w) < threshold:
        #             counter += 1

        #     if counter == 2:
        #         stop_train = True
        #         break

        # print(weight)


if __name__ == '__main__':
    main()
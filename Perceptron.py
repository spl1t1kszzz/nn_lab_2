import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def ReLU(y):
    return np.maximum(0, y)


def ReLU_derivative(z):
    if z > 0:
        return np.ones(z.shape)
    return np.zeros(z.shape)


class Perceptron:
    def __init__(self, learning_rate: float, input_dim: int, output_dim: int, hidden_sizes: [int], bias: bool):
        self.loss = 0.0
        self.weights = []
        self.biases = []
        self.outputs = []
        self.deltas = []
        self.bias = bias
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights.append(np.random.random((hidden_sizes[0], self.input_dim)))
        self.act_func = ReLU
        self.act_derivative = ReLU_derivative
        if bias:
            self.biases.append(np.zeros(hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.random((hidden_sizes[i], hidden_sizes[i - 1])))
            if bias:
                self.biases.append(np.zeros(hidden_sizes[i]))
        self.weights.append(np.random.random((output_dim, hidden_sizes[-1])))
        if bias:
            self.biases.append(np.zeros(output_dim))

    def __str__(self):
        res = ""
        for i in range(len(self.weights)):
            # if self.bias: if i == 0: res += "\tWeights " + str(self.weights[i]) + ",\tBias " + str(self.biases[i])
            # + ",\tOutputs " + str( self.outputs[i]) + "\n\n" else: res += "\tWeights " + str(self.weights[i]) + ",
            # \tBias " + str(self.biases[i]) + ",\tOutputs " + str( self.outputs[i]) + ",\tDeltas" + str(self.deltas[
            # i]) + "\n\n" else: res += "\tWeights " + str(self.weights[i]) + str(self.outputs[i]) + "\n\n" res +=
            # "\tWeights " + str(self.weights[i]) + ",\tBias " + str(self.biases[i]) + ",\tOutputs " + str(
            # self.outputs[i]) + "\n\n"
            if self.bias:
                res += "\tWeights " + str(self.weights[i]) + ',\t Bias' + str(self.biases[i]) + "\n\n"
            else:
                # res += "\tWeights " + str(self.weights[i]) +"\tDeltas " + str(self.deltas[i]) + "\n\n"
                res += "\tWeights " + str(self.weights[i]) + "\n\n"
        return res

    def forward(self, x):
        if self.bias:
            for w, b in zip(self.weights, self.biases):
                if not np.array_equal(w, self.weights[-1]):
                    x = self.act_func(w.dot(x.T).T + b)
                else:
                    x = w.dot(x.T).T + b
                self.outputs.append(x)
        else:
            for w in self.weights:
                if not np.array_equal(w, self.weights[-1]):
                    x = self.act_func(w.dot(x))
                else:
                    x = w.dot(x)
                self.outputs.append(x)
        return x

    def backward(self, error, inp):
        self.deltas.insert(0, error * 2)
        for i in reversed(range(len(self.weights) - 1)):
            layer_deltas = []
            for j in range(len(self.outputs[i])):
                delta = 0.0
                for k in range(len(self.outputs[i + 1])):
                    delta += self.weights[i + 1][k][j] * self.deltas[0][k]
                layer_deltas.append(delta * self.act_derivative(self.outputs[i][j]))
            self.deltas.insert(0, np.array(layer_deltas))
        for i in reversed(range(len(self.weights))):
            if self.bias:
                self.biases[i] -= self.learning_rate * self.deltas[i]
            for j in range(len(self.outputs[i])):
                if i == 0:
                    for k in range(len(inp)):
                        self.weights[i][j][k] -= self.learning_rate * inp[k] * self.deltas[i][j]

                else:
                    for k in range(len(self.outputs[i - 1])):
                        self.weights[i][j][k] -= self.learning_rate * self.outputs[i - 1][k] * self.deltas[i][j]

        self.deltas.clear()
        self.outputs.clear()

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            l_ = 0.0
            for x_, y_ in zip(x, y):
                out = self.forward(x_)
                error = out - y_
                self.loss = np.mean(np.square(out - y_))
                l_ += self.loss
                self.backward(error, x_)
            if (epoch + 1) % 1 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {l_}')
            # if self.loss < 1e-5:
            #     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {self.loss:.20f}')
            #     break


def clear_dataset(data, targ):
    for col in data.columns:
        if data[col].dtype == object:
            data[col], _ = pd.factorize(data[col])
        miss = np.mean(data[col].isnull())
        if miss > 0.2:
            data.drop(col, axis=1, inplace=True)
        # m = data[col].median()
        # data[col] = data[col].fillna(m)
    # data.boxplot(column=['Processor_Speed'])
    # print(data['Processor_Speed'].describe())
    # data['Storage_Capacity'].value_counts().plot.bar()
    num_rows = len(data.index)
    # too much same values
    no_inf_features = []
    for col in data.columns:
        counts = data[col].value_counts(dropna=False)
        pct_same = (counts / num_rows).iloc[0]
        if pct_same > 0.8:
            no_inf_features.append(col)
    for col in no_inf_features:
        data.drop(col, axis=1, inplace=True)
    target_corr = data.corrwith(data[targ])
    good_features = target_corr[abs(target_corr.abs()) > 0.03].index.tolist()
    return data[good_features]


if __name__ == '__main__':
    laptops = clear_dataset(pd.read_csv('Laptop_price.csv'), 'Price')
    laptops_normalized = (laptops - laptops.min()) / (laptops.max() - laptops.min())
    target = laptops_normalized['Price']
    laptops_normalized.drop(['Price'], axis=1, inplace=True)
    X_train = []
    Y_train = []
    for index, row in laptops_normalized.iterrows():
        X_train.append(np.array(row))
    for row in target:
        Y_train.append(np.array(row))
    learning_rate = 0.01
    input_size = 5
    output_size = 1
    hidden_sizes = [2, 1, 2]
    num_epochs = 10
    perceptron = Perceptron(learning_rate, input_size, output_size, hidden_sizes, True)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=190)
    perceptron.train(X_train, Y_train, num_epochs)
    perceptron_test_out = []
    perceptron_train_out = []
    for test_input in X_test:
        perceptron_test_out.append(perceptron.forward(test_input))
    for train_input in X_train:
        perceptron_train_out.append(perceptron.forward(train_input))
    print(np.mean(np.abs(np.squeeze(perceptron_test_out) - Y_test)))
    perceptron_test_out = np.squeeze(perceptron_test_out) * (laptops['Price'].max() - laptops['Price'].min()) + laptops['Price'].min()
    Y_test = np.array(Y_test) * (laptops['Price'].max() - laptops['Price'].min()) + laptops['Price'].min()
    print(np.mean(np.abs(perceptron_test_out - Y_test)))

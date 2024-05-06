import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


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
        self.act_func = sigmoid
        self.act_derivative = sigmoid_derivative
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
                    x = w.dot(x.T).T + b
                else:
                    x = self.act_func(w.dot(x.T).T + b)
                self.outputs.append(x)
        else:
            for w in self.weights:
                if not np.array_equal(w, self.weights[-1]):
                    x = w.dot(x)
                else:
                    x = self.act_func(w.dot(x))
                self.outputs.append(x)
        return x

    def backward(self, out, inp, y):
        self.deltas.insert(0, out - y)
        for i in reversed(range(len(self.weights) - 1)):
            layer_deltas = []
            for j in range(len(self.outputs[i])):
                delta = 0.0
                for k in range(len(self.outputs[i + 1])):
                    delta += self.weights[i + 1][k][j] * self.deltas[0][k]
                layer_deltas.append(delta)
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
                self.loss = -(y_ * np.log(out) + (1 - y_) * np.log(1 - out))
                l_ += self.loss
                self.backward(out, x_, y_)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {l_ / len(x)}')
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
    learning_rate = 0.01
    input_size = 17
    output_size = 1
    hidden_sizes = [1, 1, 2]
    num_epochs = 1
    target = 'poisonous'
    mushrooms = clear_dataset(pd.read_csv('mushrooms.csv'), target)
    scaler = MinMaxScaler()
    normalized_mushrooms = scaler.fit_transform(mushrooms)
    mushrooms = pd.DataFrame(normalized_mushrooms, columns=mushrooms.columns)
    mushrooms_target = mushrooms[target]
    mushrooms.drop([target], axis=1, inplace=True)
    x_train = []
    y_train = []
    for _, row in mushrooms.iterrows():
        x_train.append(row)
    for row in mushrooms_target:
        y_train.append(row)
    x_train_perc = np.array(x_train)
    y_train_perc = np.array(y_train)
    X_train, X_test, Y_train, Y_test = train_test_split(x_train_perc, y_train_perc, test_size=0.2, random_state=9)
    perceptron = Perceptron(learning_rate, input_size, output_size, hidden_sizes, True)
    perceptron.train(X_train, Y_train, 20)
    y_test_out = []
    for test_data in X_test:
        output = perceptron.forward(test_data)
        if abs(output) < abs(output - 1):
            y_test_out.append(0)
        else:
            y_test_out.append(1)
    y_test_out = np.array(y_test_out)
    tp, tn, fp, fn = 0, 0, 0, 0
    for a, b in zip(y_test_out, Y_test):
        if a == 1:
            if b == 1:
                tp += 1
            else:
                fp += 1
        else:
            if b == 0:
                tn += 1
            else:
                fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')

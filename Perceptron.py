import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def linear_func(x):
    return np.maximum(0, x)


# noinspection PyUnusedLocal
def linear_derivative(x):
    return 1


class Neuron:
    def __init__(self, activation_function, activation_derivative):
        self.val = 0.0
        self.delta = 0.0
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.links = []


class NeuralLink:
    def __init__(self, weight: float, left_neuron: Neuron, right_neuron: Neuron):
        self.weight = weight
        self.prev_delta = 0.0
        self.left_neuron = left_neuron
        self.right_neuron = right_neuron


class NeuralLayer:
    def __init__(self, links):
        self.links = links


class Perceptron:
    def __init__(self, learning_rate: float, inertial_coefficient: float, input_dim: int, output_dim: int,
                 neurons_per_layer: int,
                 number_of_hidden_layers: int):
        self.learning_rate = learning_rate
        self.inertial_coefficient = inertial_coefficient
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = []
        self.neurons_per_layer = neurons_per_layer
        self.number_of_hidden_layers = number_of_hidden_layers
        neurons = [[Neuron(linear_func, linear_derivative) for _ in range(neurons_per_layer)] for _ in
                   range(number_of_hidden_layers)]
        # Creating an input links
        input_neurons = [Neuron(linear_func, linear_derivative) for _ in range(input_dim)]
        input_links = NeuralLayer([])
        for i in range(self.input_dim):
            for j in range(neurons_per_layer):
                input_link = NeuralLink(random.random(), input_neurons[i], neurons[0][j])
                input_links.links.append(input_link)
        self.layers.append(input_links)
        # Creating an hidden links
        for i in range(number_of_hidden_layers - 1):
            layer = NeuralLayer([])
            for j in range(neurons_per_layer):
                for k in range(neurons_per_layer):
                    link = NeuralLink(random.random(), neurons[i][j], neurons[i + 1][k])
                    layer.links.append(link)
            self.layers.append(layer)
        # Creating an output links
        output_neurons = [Neuron(linear_func, linear_derivative) for _ in range(output_dim)]
        output_links = NeuralLayer([])
        for i in range(self.neurons_per_layer):
            for j in range(self.output_dim):
                output_link = NeuralLink(random.random(), neurons[self.number_of_hidden_layers - 1][i],
                                         output_neurons[j])
                output_links.links.append(output_link)
        self.layers.append(output_links)

    def __str__(self):
        result = ""
        for i in range(self.number_of_hidden_layers + 1):
            if i == 0:
                result += f'Layer {i} (input) :\n'
            elif i == self.number_of_hidden_layers:
                result += f'Layer {i} (output) :\n'
            else:
                result += f'Layer {i} :\n'
            for link in self.layers[i].links:
                result += (f'\t Weight {link.weight}, Left neuron: {link.left_neuron.val},'
                           f' Delta: {link.left_neuron.delta},'
                           f' Right neuron: {link.right_neuron.val},'
                           f' Delta  {link.right_neuron.delta}\n')
        return result

    def predict(self, x_data: np.ndarray):
        for i in range(self.input_dim):
            for j in range(self.neurons_per_layer):
                self.layers[0].links[i * self.neurons_per_layer + j].left_neuron.val = x_data[i]
        for i in range(self.number_of_hidden_layers + 1):
            for link in self.layers[i].links:
                link.right_neuron.val += link.left_neuron.val * link.weight
        result = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            neuron = self.layers[self.number_of_hidden_layers].links[i].right_neuron
            result[i] = neuron.activation_function(neuron.val)
        return result

    def train(self, x_data: np.ndarray, y_data: np.ndarray, epochs: int):
        for epoch in range(epochs):
            total_loss = 0.0
            for x_, y_ in zip(x_data, y_data):
                output = self.predict(x_)
                error = y_ - output
                for i in range(self.output_dim):
                    output_neuron = self.layers[self.number_of_hidden_layers].links[i].right_neuron
                    output_neuron.delta = error[i] * output_neuron.activation_derivative(output_neuron.val)
                for j in reversed(range(self.number_of_hidden_layers + 1)):
                    for link in self.layers[j].links:
                        link.left_neuron.delta += (link.right_neuron.delta * link.weight
                                                   * link.left_neuron.activation_derivative(link.left_neuron.val))
                for k in range(self.number_of_hidden_layers + 1):
                    for link in self.layers[k].links:
                        delta_weight = (self.learning_rate * link.left_neuron.val * link.right_neuron.delta +
                                        self.inertial_coefficient * link.prev_delta)
                        link.prev_delta = delta_weight
                        link.weight += delta_weight
                for l in range(self.number_of_hidden_layers + 1):
                    for link in self.layers[l].links:
                        if l != 0:
                            link.left_neuron.val = 0.0
                        link.right_neuron.val = 0.0
                        link.left_neuron.delta = 0.0
                        link.right_neuron.delta = 0.0
                loss = np.mean(np.square(y_ - output))
                total_loss += loss
            avg_loss = total_loss / len(x_data)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    perceptron = Perceptron(0.000001, 0.00001, 1, 1, 2, 2)
    x = np.array([[1], [2], [4], [5], [7], [9], [10], [11], [12], [13], [14], [15]])
    y = np.array([[2], [4], [8], [10], [14], [18], [20], [22], [24], [26], [28], [30]])
    perceptron.train(x, y, 1000)
    print(perceptron.predict(np.array([2001])))


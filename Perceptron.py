import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def linear_func(x):
    return x


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
        self.left_neuron = left_neuron
        self.right_neuron = right_neuron


class NeuralLayer:
    def __init__(self, links):
        self.links = links


class Perceptron:
    def __init__(self, input_dim: int, output_dim: int, neurons_per_layer: int, number_of_hidden_layers: int):
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

    def predict(self, x_data: np.ndarray):
        for i in range(self.input_dim):
            for j in range(self.neurons_per_layer):
                self.layers[0].links[i * self.neurons_per_layer + j].left_neuron.val = x_data[i]
        for i in range(self.number_of_hidden_layers + 1):
            for link in self.layers[i].links:
                link.right_neuron.val += link.left_neuron.val * link.weight
            for link in self.layers[i].links:
                link.right_neuron.val = link.right_neuron.activation_function(link.right_neuron.val)
        result = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            result[i] = self.layers[self.number_of_hidden_layers].links[i].right_neuron.val
        return result

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

    def train(self, x_data: np.ndarray, y_data: np.ndarray, epochs: int):
        for i in range(epochs):
            out = self.predict(x_data)
            error = y_data - out
            for j in range(self.output_dim):
                # Calculating delta for output layer
                output_neuron = self.layers[self.number_of_hidden_layers].links[j].right_neuron
                output_neuron.delta = error[j] * output_neuron.activation_derivative(output_neuron.val)
                # Calculating delta for hidden layers
            for k in reversed(range(self.number_of_hidden_layers + 1)):
                for m in range(len(self.layers[k].links)):
                    link = self.layers[k].links[m]
                    link.left_neuron.delta += (link.right_neuron.delta * link.weight *
                                               link.left_neuron.activation_derivative(link.left_neuron.val))
                    m += self.neurons_per_layer


if __name__ == '__main__':
    perceptron = Perceptron(1, 3, 2, 2)
    perceptron.train(np.array([1]), np.array([2, 3, 4]), 1)
    print(perceptron)
    # for i in range(perceptron.output_dim):
    #     print(perceptron.layers[perceptron.number_of_hidden_layers].links[i].right_neuron.val)
    # for link in perceptron.layers[perceptron.number_of_hidden_layers].links:
    #     print(f'Weight {link.weight}, Left neuron: {link.left_neuron.val},'
    #                        f' Right neuron: {link.right_neuron.val}\n')

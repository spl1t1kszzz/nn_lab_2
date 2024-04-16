import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, val, activation_function):
        self.val = val
        self.activation_function = activation_function
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
        neurons = [[Neuron(0.0, sigmoid) for _ in range(neurons_per_layer)] for _ in range(number_of_hidden_layers)]
        # Creating an input links
        input_neurons = [Neuron(0.0, sigmoid) for _ in range(input_dim)]
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
        output_neurons = [Neuron(0.0, sigmoid) for _ in range(output_dim)]
        output_links = NeuralLayer([])
        for i in range(self.neurons_per_layer):
            for j in range(self.output_dim):
                output_link = NeuralLink(random.random(), neurons[self.number_of_hidden_layers - 1][i],
                                         output_neurons[j])
                output_links.links.append(output_link)
        self.layers.append(output_links)

    def predict(self):
        for i in range(self.number_of_hidden_layers + 1):
            for link in self.layers[i].links:
                link.right_neuron.val += link.left_neuron.val * link.weight

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
                           f' Right neuron: {link.right_neuron.val}\n')
        return result


if __name__ == '__main__':
    perceptron = Perceptron(2, 1, 2, 2)
    perceptron.layers[0].links[0].left_neuron.val = 1
    perceptron.layers[0].links[1].left_neuron.val = 1
    perceptron.layers[0].links[2].left_neuron.val = 2
    perceptron.layers[0].links[3].left_neuron.val = 2
    print(perceptron)

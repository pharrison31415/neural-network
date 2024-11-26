import random
from pprint import pprint as pp
import numpy as np


class Network:
    def __init__(self, *shape):
        self.shape = shape
        self.layer_count = len(shape)
        self.weights = [
            np.random.rand(curr, prev)
            for prev, curr in zip(self.shape, self.shape[1:])
        ]

        self.biases = [
            np.random.uniform(low=-prev, high=prev, size=(curr))
            for prev, curr in zip(self.shape, self.shape[1:])
        ]

        self.activations = [
            np.zeros(neuron_count)
            for neuron_count in self.shape
        ]

    def sigmoid(self, x):
        return (np.exp(-x) + 1) ** -1

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, x):
        self.activations[0] = x
        for layer in range(self.layer_count-1):  # Hidden layers onward
            w = self.weights[layer]
            b = self.biases[layer]
            self.activations[layer + 1] = self.sigmoid(
                np.matmul(w, self.activations[layer]) + b)

    def backpropagate(self, desired_y, learn_rate):
        deltas = [np.zeros(neuron_count) for neuron_count in self.shape]

        # Calculate error and delta for the output layer
        output_error = desired_y - self.activations[-1]
        output_delta = output_error * \
            self.sigmoid_derivative(self.activations[-1])
        deltas[-1] = output_delta

        # Backpropagate through the hidden layers
        for layer in range(self.layer_count - 2, 0, -1):
            error = np.matmul(self.weights[layer].T, deltas[layer + 1])
            delta = error * self.sigmoid_derivative(self.activations[layer])
            deltas[layer] = delta

        # Update weights and biases
        for layer in range(self.layer_count - 1):
            self.weights[layer] += learn_rate * \
                np.outer(deltas[layer + 1], self.activations[layer])
            self.biases[layer] += learn_rate * deltas[layer + 1]

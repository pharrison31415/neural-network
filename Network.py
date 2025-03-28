import numpy as np


def sigmoid(z):
    return (np.exp(-z) + 1) ** -1


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(z > 0, 1, 0)


def mean_squared_error(a, y):
    return np.mean((y - a)**2)


def mean_squared_error_derivative(a, y):
    return (a - y) * 2


class Network:
    def __init__(self, *shape):
        self.shape = shape
        self.layer_count = len(shape)

        self.w = [0] + [  # Sentinel weight, as input layer has no weights
            np.random.rand(curr, prev)
            for prev, curr in zip(self.shape, self.shape[1:])
        ]

        self.b = [0] + [  # Sentinel bias, as input layer has no bias
            np.random.uniform(low=-prev, high=prev, size=(curr))
            for prev, curr in zip(self.shape, self.shape[1:])
        ]

        self.z = [np.zeros(neuron_count) for neuron_count in self.shape]
        self.a = [np.zeros(neuron_count) for neuron_count in self.shape]

    def activation(self, z):
        return sigmoid(z)

    def activation_derivative(self, z):
        return sigmoid_derivative(z)

    def cost(self, a, y):
        return mean_squared_error(a, y)

    def cost_derivative(self, a, y):
        return mean_squared_error_derivative(a, y)

    def train(self, x_arr, y_arr, learn_rate):
        batch_nabla_w = []
        batch_nabla_b = []
        for x, y in zip(x_arr, y_arr):
            self.feedforward(x)
            nabla_w, nabla_b = self.backpropagate(y)
            batch_nabla_w.append(nabla_w)
            batch_nabla_b.append(nabla_b)

        avg_nabla_w = []
        for nw in zip(*batch_nabla_w):
            avg_nabla_w.append(np.average(nw, axis=0))
        self.update_weights(avg_nabla_w, learn_rate)

        avg_nabla_b = []
        for nb in zip(*batch_nabla_b):
            avg_nabla_b.append(np.average(nb, axis=0))
        self.update_biases(avg_nabla_b, learn_rate)

    def feedforward(self, x):
        assert x.shape == self.a[0].shape

        self.a[0] = x
        for l in range(1, self.layer_count):  # Hidden layers onward
            z = np.dot(self.w[l], self.a[l - 1]) + self.b[l]
            a = self.activation(z)

            self.z[l] = z
            self.a[l] = a

    def backpropagate(self, y):
        nabla_w = [np.zeros_like(w) for w in self.w]
        nabla_b = [np.zeros_like(b) for b in self.b]

        for l in range(self.layer_count - 1, 0, -1):
            dC_da = self.cost_derivative(self.a[l], y)
            da_dz = self.activation_derivative(self.z[l])
            dC_dz = dC_da * da_dz

            dz_dw = self.a[l-1]
            dC_dw = np.outer(dC_dz, dz_dw)
            nabla_w[l] = dC_dw

            nabla_b[l] = dC_dz   # dC/db = dC/dz * dz/db; but dz/db = 1

        # Sentinel nabla weight and nabla bias
        return [0] + nabla_w, [0] + nabla_b

    def update_weights(self, nabla_w, learn_rate):
        for l in range(self.layer_count - 1):
            self.w[l] += learn_rate * nabla_w[l]

    def update_biases(self, nabla_b, learn_rate):
        for l in range(self.layer_count - 1):
            self.b[l] += learn_rate * nabla_b[l]

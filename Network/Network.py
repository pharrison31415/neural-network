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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class Network:
    def __init__(self, shape, activation_fn="sigmoid"):
        self.shape = shape
        self.layer_count = len(shape)
        self.activation_fn = activation_fn

        # Weights
        self.w = [
            np.random.randn(curr, prev)
            for prev, curr in zip(self.shape[:-1], self.shape[1:])
        ]

        # Biases
        self.b = [np.random.randn(curr) for curr in self.shape[1:]]

        # Z-vectors; pre-activation values
        self.z = [np.zeros(neuron_count) for neuron_count in self.shape]
        # Activation values
        self.a = [np.zeros(neuron_count) for neuron_count in self.shape]

    def activation(self, z):
        if self.activation_fn == "sigmoid":
            return sigmoid(z)
        elif self.activation_fn == "relu":
            return relu(z)
        elif self.activation_fn == "tanh":
            return np.tanh(z)

    def activation_derivative(self, z):
        if self.activation_fn == "sigmoid":
            return sigmoid_derivative(z)
        elif self.activation_fn == "relu":
            return relu_derivative(z)
        elif self.activation_fn == "tanh":
            # Hyperbolic secant squared
            return (1 / np.cosh(z)) ** 2

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
        for l in range(self.layer_count - 1):
            z = np.dot(self.w[l], self.a[l]) + self.b[l]
            a = self.activation(z)

            self.z[l+1] = z
            self.a[l+1] = a

    def evaluate(self, x, apply_softmax=False, sort=False):
        self.feedforward(x)
        out = np.copy(self.a[-1])

        # Normalize such that Σ(out) = 1
        if apply_softmax:
            out = softmax(out)

        if sort:
            # Enumerate each element
            out = np.column_stack((np.arange(out.shape[0]), out))
            # Sort by second column
            out = out[out[:, 1].argsort()]

        out.flags.writeable = False
        return out

    def backpropagate(self, y):
        nabla_w = [np.zeros_like(w) for w in self.w]
        nabla_b = [np.zeros_like(b) for b in self.b]

        for l in range(self.layer_count - 2, -1, -1):
            if l == self.layer_count - 2:
                # Use y for desired activation values
                dC_da = self.cost_derivative(self.a[l+1], y)
            else:
                # dC/da[l] = dz[l+1]/da[l] * da[l+1]/dz[l+1] * dC/da[l+1]
                # dz[l+1]/da[l] = w[l+1]
                # da[l+1]/dz[l+1] * dC/da[l+1] = dC/dz[l+1]
                # dC/da[l] = w[l+1] * dC/dz[l+1]
                dC_da = np.dot(self.w[l+1].transpose(), dC_dz)

            # dC/dz = dC/da * da/dz;  da/dz = σ'(z)
            dC_dz = dC_da * self.activation_derivative(self.z[l+1])

            # dC/db = dC/dz * dz/db;  dz/db = 1
            nabla_b[l] = dC_dz

            # dC/dw = dC/dz * dz_dw;  dz/dw = a[l]
            nabla_w[l] = np.outer(dC_dz, self.a[l])

        return nabla_w, nabla_b

    def update_weights(self, nabla_w, learn_rate):
        for l in range(self.layer_count - 1):
            self.w[l] -= learn_rate * nabla_w[l]

    def update_biases(self, nabla_b, learn_rate):
        for l in range(self.layer_count - 1):
            self.b[l] -= learn_rate * nabla_b[l]

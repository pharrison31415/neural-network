from Network import Network
import numpy as np
import random

# Initialize network
net = Network(2, 3, 1)

# Train network for 10_000 epochs of mini-batches of size 10
for batch in range(10_000):
    x_arr = []
    y_arr = []
    for _ in range(10):
        x1 = random.choice([0.0, 1.0])
        x2 = random.choice([0.0, 1.0])
        y = float(x1 != x2)  # y = x1 XOR x2

        x_arr.append(np.array([x1, x2]))
        y_arr.append(np.array([y]))

    net.train(x_arr, y_arr, 1.0)

# Demonstrate results of XOR training
domain = (
    np.array([0.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([1.0, 0.0]),
    np.array([1.0, 1.0]),
)
for x in domain:
    net.feedforward(x)
    y = net.a[-1]
    print(f"{x} -> {y}")

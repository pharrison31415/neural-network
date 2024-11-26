from Network import Network
import numpy as np
import random

# Initialize network
net = Network(2, 3, 1)

# Train network for 10_0000 epochs to perform xor
for epoch in range(10_000):
    x1 = random.choice([0.0, 1.0])
    x2 = random.choice([0.0, 1.0])
    y = float(x1 and not x2 or not x1 and x2)
    net.feedforward(np.array([x1, x2]))
    net.backpropagate(np.array([y]), 1.0)

# Demonstrate results of xor training
domain = (
    np.array([0.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([1.0, 0.0]),
    np.array([1.0, 1.0]),
)
for x in domain:
    net.feedforward(x)
    y = net.activations[-1]
    print(f"{x} -> {y}")

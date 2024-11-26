from Network import Network
import numpy as np
import random

# Initialize network
net = Network(1, 2, 1)

# Train network for 10_0000 epochs to negate input
for epoch in range(10_000):
    x = random.choice([0.0, 1.0])
    y = float(not x)
    net.feedforward(np.array([x]))
    net.backpropagate(np.array([y]), 1.0)

# Demonstrate results of negation training
for x in (np.array([0.0]), np.array([1.0])):
    net.feedforward(x)
    y = net.activations[-1]
    print(f"{x} -> {y}")

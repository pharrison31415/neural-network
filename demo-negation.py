from Network import Network
import numpy as np
import random

# Initialize network
net = Network(1, 2, 1)

# Train network for 10_0000 epochs to negate input
for batch in range(100):
    x_list = np.array([[random.choice([0.0, 1.0])] for _ in range(100)])
    y_list = np.array([[float(not x[0])] for x in x_list])
    net.train_batch(x_list, y_list, 0.1)

# Demonstrate results of negation training
for x in (np.array([0.0]), np.array([1.0])):
    net.feedforward(x)
    y = net.a[-1]
    print(f"{x} -> {y}")

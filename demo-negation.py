from Network import Network
import numpy as np
import random

# Initialize network
net = Network(1, 2, 1)

# Train network for 1000 epochs of batches of size 10
for batch in range(1000):
    x_arr = np.array([[random.choice([0.0, 1.0])] for _ in range(10)])
    y_arr = np.array([[float(not x[0])] for x in x_arr])  # y = NOT x
    net.train_batch(x_arr, y_arr, 0.1)

# Demonstrate results of negation training
for x in (np.array([0.0]), np.array([1.0])):
    net.feedforward(x)
    y = net.a[-1]
    print(f"{x} -> {y}")

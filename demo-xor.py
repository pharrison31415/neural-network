from Network import Network
import numpy as np
import random

# Initialize network
net = Network(2, 3, 1)

# Train network for 10_0000 epochs to perform xor
for batch in range(1000):
    x_list = []
    y_list = []
    for _ in range(10):
        x1 = random.choice([0.0, 1.0])
        x2 = random.choice([0.0, 1.0])
        y = float(x1 and not x2 or not x1 and x2)

        x_list.append(np.array([x1, x2]))
        y_list.append(np.array([y]))

    net.learn_batch(x_list, y_list, 0.1)

# Demonstrate results of xor training
domain = (
    np.array([0.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([1.0, 0.0]),
    np.array([1.0, 1.0]),
)
for x_list in domain:
    net.feedforward(x_list)
    y_list = net.a[-1]
    print(f"{x_list} -> {y_list}")

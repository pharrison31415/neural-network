import os
import pickle
import sys

import numpy as np
import pandas as pd

from Network.Network import Network

HELP_MESSAGE = """train-mnsit.py Usage:

$ python train-mnist.py MNIST_CSV OUT_FILE

 - MNIST_CSV is a csv file containing labels and pixel values for the MNIST dataset. Try unzipping mnist.csv.zip.
 - OUT_FILE is a file to dump the contents of the trained network.
"""


def print_progress_bar(iteration, total, bar_length=40):
    progress = iteration / total
    arrow = '=' * int(round(progress * bar_length) - 1)
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write(
        f'\r[{arrow}{spaces}] {progress * 100:.2f}% Epoch {iteration} of {total}')
    sys.stdout.flush()


def train_network(df, epochs, mini_batch_size, learn_rate, progress_bar=True):
    # Initialize network
    net = Network([28*28, 16, 16, 10])

    # Training
    for epoch in range(epochs):
        # Print progress bar if desired
        if progress_bar:
            print_progress_bar(epoch, epochs)

        # Build mini-batch
        mini_batch = df.sample(n=mini_batch_size, replace=True)

        # Regex for [string of digits]x[string of digits]
        # Columns for pixel values are of the form 1x1, 1x2, ... 28x28
        pixel_column_regex = '^\d+x\d+$'

        # Grab the colums holding pixel values
        batch_pixels = mini_batch.filter(regex=pixel_column_regex, axis=1)

        # Mini-batch x values
        x_arr = np.array(batch_pixels.astype(np.float64) / 256)

        # Mini-batch y values
        y_lst = []
        batch_labels = np.array(mini_batch["label"])
        for label in batch_labels:
            y = [0.0] * 10
            y[label] = 1.0
            y_lst.append(y)
        y_arr = np.array(y_lst)

        # Train network on mini-batch
        net.train(x_arr, y_arr, learn_rate)

    # Newline to keep the progress bar pretty
    if progress_bar:
        print()

    # Return trained network
    return net


if __name__ == "__main__":
    # Validate length of arguments
    if len(sys.argv) != 3:
        print(HELP_MESSAGE, file=sys.stderr)
        sys.exit(1)

    # Grab system arguments
    IN_CSV = sys.argv[1]
    OUT_FILE = sys.argv[2]

    # Check if IN_CSV file exists
    if not os.path.exists(IN_CSV):
        print(f"Error: The file \"{IN_CSV}\" does not exist.", file=sys.stderr)
        sys.exit(1)

    # Read MNIST csv
    mnist_df = pd.read_csv(IN_CSV)

    # Train network
    net = train_network(mnist_df, epochs=10_000,
                        mini_batch_size=100, learn_rate=0.1)

    # Save pickle dump of network
    with open(OUT_FILE, "wb") as file:
        pickle.dump(net, file)

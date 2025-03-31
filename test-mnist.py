import os
import pickle
import sys

import numpy as np
import pandas as pd

HELP_MESSAGE = """train-mnsit.py Usage:

$ python test-mnist.py MNIST_CSV NETWORK_FILE

 - MNIST_CSV is a csv file containing labels and pixel values for the MNIST dataset. Try unzipping mnist.csv.zip.
 - NETWORK_FILE is a binary pickle dump of a Network object. Try mnist.network
"""


def test_network(network, df, shots=1000):
    correct = 0
    incorrect = 0
    for n in range(shots):
        # Get a sample
        sample = df.sample(n=1).reset_index()

        # Regex for [string of digits]x[string of digits]
        # Columns for pixel values are of the form 1x1, 1x2, ... 28x28
        pixel_column_regex = '^\d+x\d+$'

        # Grab the colums holding pixel values
        pixels = sample.filter(regex=pixel_column_regex, axis=1).loc[0]

        # Build x
        x = np.array(pixels.astype(np.float64) / 256)

        # Evaluate network
        network.feedforward(x)
        activations = list(network.a[-1])
        guess = activations.index(max(activations))

        # Score
        if guess == sample.loc[0, "label"]:
            correct += 1
        else:
            incorrect += 1

    return shots, correct, incorrect


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(HELP_MESSAGE, file=sys.stderr)
        sys.exit(1)

    # Grab system arguments
    IN_CSV = sys.argv[1]
    NETWORK_FILE = sys.argv[2]

    # Check if IN_CSV file exists
    if not os.path.exists(IN_CSV):
        print(f"Error: The file \"{IN_CSV}\" does not exist.", file=sys.stderr)
        sys.exit(1)

    # Read MNIST csv
    mnist_df = pd.read_csv(IN_CSV)

    # Load Network object
    with open(NETWORK_FILE, "rb") as file:
        network = pickle.load(file)

    total, correct, incorrect = test_network(network, mnist_df, shots=10000)
    print("\t\tCount\tPercentage")
    print(f"Correct:\t{correct}\t{correct/total*100}%")
    print(f"Incorrect:\t{incorrect}\t{incorrect/total*100}%")
    print(f"Total:\t\t{total}")

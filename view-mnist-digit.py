import os
import sys

import numpy as np
import pandas as pd
import pygame


HELP_MESSAGE = """view-mnist-digit.py Usage:

$ python view-mnist-digit.py MNIST_CSV INDEX

 - MNIST_CSV is a csv file containing labels and pixel values for the MNIST dataset. Try unzipping mnist.csv.zip.
 - INDEX is the number to initially display

Press the left and right arrows to navigate through the dataset
"""

GRID_SIZE = 28
CELL_SIZE = 24
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

WHITE = 255
BLACK = 0


def get_canvas(df, index):
    return np.array(df.loc[index]).reshape((28, 28))


def draw_canvas(screen, canvas):
    for r in range(canvas.shape[0]):
        for c in range(canvas.shape[1]):
            pygame.draw.rect(
                screen, [canvas[r][c]] * 3, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))


def display_digit(df, initial_index):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("View an MNIST digit")

    canvas = np.array([[BLACK] * GRID_SIZE] * GRID_SIZE, dtype=np.uint8)

    running = True
    index = initial_index
    canvas = get_canvas(df, index)
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            # Handle quitting
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # Handle next index
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                index = (index + 1) % len(df)
                canvas = get_canvas(df, index)
            # Handle previous index
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                index = (index - 1) % len(df)
                canvas = get_canvas(df, index)

        # Update display
        draw_canvas(screen, canvas)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    # Validate length of arguments
    if len(sys.argv) != 3:
        print(HELP_MESSAGE, file=sys.stderr)
        sys.exit(1)

    # Grab system arguments
    MNIST_CSV = sys.argv[1]
    INDEX_STR = sys.argv[2]

    # Check if IN_CSV file exists
    if not os.path.exists(MNIST_CSV):
        print(
            f"Error: The file \"{MNIST_CSV}\" does not exist.", file=sys.stderr)
        sys.exit(1)

    # Validate INDEX argument
    try:
        index = int(INDEX_STR)
    except ValueError:
        print("Error: INDEX must be a positive integer")

    # Read MNIST csv
    mnist_data = pd.read_csv(MNIST_CSV)

    # Regex for [string of digits]x[string of digits]
    # Columns for pixel values are of the form 1x1, 1x2, ... 28x28
    pixel_column_regex = '^\d+x\d+$'

    # Grab the colums holding pixel values
    filtered_df = mnist_data.filter(regex=pixel_column_regex, axis=1)

    display_digit(filtered_df, index)

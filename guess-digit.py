import os
import pickle
import sys

import numpy as np
import pygame


HELP_MESSAGE = """guess-digit.py Usage:

$ python guess-digit.py NETWORK_FILE

 - NETWORK_FILE is a binary pickle dump of a Network object. Try mnist.network

Click and drag to draw a digit. Hit space to clear canvas.
"""

# Validate length of arguments
if len(sys.argv) != 2:
    print(HELP_MESSAGE, file=sys.stderr)
    sys.exit(1)

# Grab and validate network file argument
NETWORK_FILE = sys.argv[1]
if not os.path.exists(NETWORK_FILE):
    print(
        f"Error: The file \"{NETWORK_FILE}\" does not exist.", file=sys.stderr)
    sys.exit(1)

# Load Network object
with open(NETWORK_FILE, "rb") as file:
    network = pickle.load(file, fix_imports=False)


GRID_SIZE = 28
CELL_SIZE = 24
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

WHITE = 255
BLACK = 0

PEN_RADIUS = 2 * CELL_SIZE
PEN_INTENSITY = 0.2

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a digit")

canvas = np.array([[BLACK] * GRID_SIZE] * GRID_SIZE, dtype=np.uint8)


def get_intensity(distance, radius):
    return np.where(distance > radius, 0,  (1 - distance / radius)) * PEN_INTENSITY


def draw_canvas(screen, canvas):
    for r in range(canvas.shape[0]):
        for c in range(canvas.shape[1]):
            pygame.draw.rect(
                screen, [canvas[r][c]] * 3, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))


running = True
pen_down = False
mouse_move = False
old_mouse_x, old_mouse_y = -1, -1

while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        # Handle quitting
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

        # Handle pen up/down
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pen_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                pen_down = False

        # Handle clear canvas with space
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            canvas = np.array([[BLACK] * GRID_SIZE] *
                              GRID_SIZE, dtype=np.uint8)

    # Get mouse position in cell coordinates
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if (mouse_x, mouse_y) != (old_mouse_x, old_mouse_y):
        mouse_move = True
    mouse_col = mouse_x // CELL_SIZE
    mouse_row = mouse_y // CELL_SIZE

    if pen_down and mouse_move:
        # Create array of distances to mouse
        distance = np.zeros_like(canvas, dtype=np.float64)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                center_x = c * CELL_SIZE + CELL_SIZE // 2
                center_y = r * CELL_SIZE + CELL_SIZE // 2
                distance[r][c] = np.sqrt((center_x - mouse_x)
                                         ** 2 + (center_y - mouse_y) ** 2)

        # Calculate the intensity based on the distance and pen radius
        intensity_arr = get_intensity(distance, PEN_RADIUS)

        # Update canvas
        overflow_canvas = np.add(
            canvas, (WHITE * intensity_arr).astype(np.uint8), dtype=np.uint16)
        canvas = np.clip(overflow_canvas, a_min=0, a_max=255)

        # Guess digit
        x = canvas.flatten()/255
        network.feedforward(x)
        probabilities = list(network.evaluate(x, do_softmax=True))
        best_guess = probabilities.index(max(probabilities))
        print(f"{best_guess} -> {probabilities[best_guess]}")

        old_mouse_x = mouse_x
        old_mouse_y = mouse_y

    # Update display
    draw_canvas(screen, canvas)
    pygame.draw.circle(screen, "gray", (mouse_x, mouse_y), PEN_RADIUS, 1)
    pygame.display.flip()

pygame.quit()

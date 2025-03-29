import sys
import pandas as pd
from PIL import Image


def view_mnist_digit(mnist_data, index):
    img = Image.new(mode="L", size=[28, 28])

    for y in range(img.size[0]):
        for x in range(img.size[1]):
            color = int(mnist_data[f"{y+1}x{x+1}"][index])
            img.putpixel([x, y], color)

    img.show()


if __name__ == "__main__":
    mnist_path = sys.argv[1]
    index = int(sys.argv[2])

    mnist_data = pd.read_csv(mnist_path)
    view_mnist_digit(mnist_data, index)

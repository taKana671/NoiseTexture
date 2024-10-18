import random

import cv2
import numpy as np

from noise import Noise


class Cellular(Noise):

    def __init__(self, grid=4, size=256):
        super().__init__()
        self.size = size
        self.grid = grid

    def fdist2(self, x, y):
        p = np.array([x, y])
        n = np.floor(p + 0.5)
        dist = 2.0 ** 0.5

        for j in range(3):
            grid = np.zeros(2)
            grid[1] = n[1] + np.sign(j % 2 - .5) * np.ceil(j * .5)

            if abs((grid - p)[1]) - 0.5 > dist:
                continue

            for i in range(-1, 2):
                grid[0] = n[0] + i
                jitter = self.hash22(grid) - 0.5
                vec = grid + jitter - p
                length = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
                dist = min(dist, length)

        return dist

    def fdist3(self, x, y, t):
        p = np.array([x, y, t])
        n = np.floor(p + 0.5)
        dist = 3.0 ** 0.5

        for k in range(3):
            grid = np.zeros(3)
            grid[2] = n[2] + np.sign(k % 2. - .5) * np.ceil(k * .5)

            if abs((grid - p)[2]) - 0.5 > dist:
                continue

            for j in range(3):
                grid[1] = n[1] + np.sign(j % 2 - .5) * np.ceil(j * .5)

                if abs((grid - p)[1]) - 0.5 > dist:
                    continue

                for i in range(-1, 2):
                    grid[0] = n[0] + i
                    jitter = self.hash33(grid) - 0.5
                    v = grid + jitter - p
                    length = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
                    dist = min(dist, length)

        return dist

    def noise2(self, t=None):
        if t is None:
            t = random.uniform(0, 1000)

        self.hash = {}

        arr = np.array(
            [self.fdist2(x + t, y + t) for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def noise3(self, t=None):
        if t is None:
            t = random.uniform(0, 1000)

        self.hash = {}

        arr = np.array(
            [self.fdist3(x + t, y + t, t) for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr


def create_img_8bit(path, grid=4, size=256):
    cellular = Cellular(grid, size)
    # arr = cellular.noise3()
    arr = cellular.noise2()

    arr *= 255
    arr = arr.astype(np.uint8)
    cv2.imwrite(path, arr)


def create_img_16bit(path, grid=4, size=256):
    cellular = Cellular(grid, size)
    # arr = cellular.noise3()
    arr = cellular.noise2()
    # arr = np.abs(arr)
    arr *= 65535
    arr = arr.astype(np.uint16)
    cv2.imwrite(path, arr)


if __name__ == '__main__':
    create_img_8bit('cellular_sample01.png')
    # create_img_16bit('cellular_sample01.png')


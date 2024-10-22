import cv2
import numpy as np

from noise import Noise


class Periodic(Noise):
    """Add periodicity to random numbers to create a periodic noise
       which ends are connnected.
    """

    def __init__(self, period=4, grid=4, size=256):
        self.period = period
        self.size = size
        self.grid = grid

    def periodic2(self, p):
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable2(np.mod(n + (arr := np.array([i, j])), self.period), f - arr)
             for j in range(2) for i in range(2)]

        f = 6 * f**5 - 15 * f**4 + 10 * f**3
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def periodic3(self, p):
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable3(np.mod(n + (arr := np.array([i, j, k])), self.period), f - arr) * 0.70710678
             for k in range(2) for j in range(2) for i in range(2)]

        f = 6 * f**5 - 15 * f**4 + 10 * f**3
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def noise2(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        half_period = self.period / 2
        half_grid = self.grid / 2
        arr = np.zeros((self.size, self.size))

        for j, y in enumerate(np.linspace(0, self.grid, self.size)):
            for i, x in enumerate(np.linspace(0, self.grid, self.size)):
                px, py = self.xy2pol(x - half_grid, y - half_grid)
                _x = half_period / np.pi * px + t
                _y = half_period * py
                arr[i, j] = self.periodic2(np.array([_x, _y]))

        return arr

    def noise3(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        half_period = self.period / 2
        half_grid = self.grid / 2
        arr = np.zeros((self.size, self.size))

        for j, y in enumerate(np.linspace(0, self.grid, self.size)):
            for i, x in enumerate(np.linspace(0, self.grid, self.size)):
                px, py = self.xy2pol(x - half_grid, y - half_grid)
                _x = half_period / np.pi * px + t
                _y = half_period * py + t
                arr[i, j] = self.periodic3(np.array([_x, _y, t]))

        return arr


# np.count_nonzero(np.sign(arr) < 0) ; no less than zero: no
def create_img_8bit(path, period=4, grid=4, size=256):
    periodic = Periodic(period, grid, size)
    arr = periodic.noise3()

    arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
    # arr *= 255
    # arr = arr.astype(np.uint8)
    cv2.imwrite(path, arr)


def create_img_16bit(path, period=4, grid=4, size=256):
    periodic = Periodic(period, grid, size)
    arr = periodic.noise2()

    arr *= 65535
    arr = arr.astype(np.uint16)
    cv2.imwrite(path, arr)


if __name__ == '__main__':
    create_img_8bit('periodic_sample02.png')
    # create_img_16bit('periodic_sample02.png')
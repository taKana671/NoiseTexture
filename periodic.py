import random

import cv2
import numpy as np

from noise import Noise


class Periodic(Noise):

    def __init__(self, period=4, grid=4, size=256):
        super().__init__()
        self.period = period
        self.size = size
        self.grid = grid

    def gtable2(self, lattice, p):
        lattice = lattice.astype(np.uint)

        if (tup := tuple(lattice)) in self.hash:
            idx = self.hash[tup]
        else:
            idx = random.randint(1, 6)
            self.hash[tup] = idx

        u = (p[0] if idx < 4 else p[1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[1] if idx < 4 else p[0]) * 0.38268343   # 0.38268343 = sin(pi/8)

        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    def gtable3(self, lattice, p):
        lattice = lattice.astype(np.uint)

        if (tup := tuple(lattice)) in self.hash:
            idx = self.hash[tup]
        else:
            idx = random.randint(0, 15)
            self.hash[tup] = idx

        u = p[0] if idx < 8 else p[1]

        if idx < 4:
            v = p[1]
        elif idx == 12 or idx == 14:
            v = p[0]
        else:
            v = p[2]

        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    def periodic2(self, x, y):
        p = np.array([x, y])
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable2(np.mod(n + (arr := np.array([i, j])), self.period), f - arr)
             for j in range(2) for i in range(2)]

        f = 6 * f**5 - 15 * f**4 + 10 * f**3
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def periodic3(self, x, y, t):
        p = np.array([x, y, t])
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable3(np.mod(n + (arr := np.array([i, j, k])), self.period), f - arr) * 0.70710678
             for k in range(2) for j in range(2) for i in range(2)]

        f = 6 * f**5 - 15 * f**4 + 10 * f**3
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def noise2(self, t=None):
        if t is None:
            t = random.uniform(0, 1000)

        self.hash = {}
        half_period = self.period / 2
        half_grid = self.grid / 2
        arr = np.zeros((self.size, self.size))

        for j, y in enumerate(np.linspace(0, self.grid, self.size)):
            for i, x in enumerate(np.linspace(0, self.grid, self.size)):
                _x, _y = self.xy2pol(x - half_grid, y - half_grid)
                arr[i, j] = self.periodic2(half_period / np.pi * _x + t, half_period * _y + t)

        return arr

    def noise3(self, t=None):
        if t is None:
            t = random.uniform(0, 1000)

        self.hash = {}
        half_period = self.period / 2
        half_grid = self.grid / 2
        arr = np.zeros((self.size, self.size))

        for j, y in enumerate(np.linspace(0, self.grid, self.size)):
            for i, x in enumerate(np.linspace(0, self.grid, self.size)):
                _x, _y = self.xy2pol(x - half_grid, y - half_grid)
                arr[i, j] = self.periodic3(half_period / np.pi * _x + t, half_period * _y + t, t)

        return arr


# np.count_nonzero(np.sign(arr) < 0) ; no less than zero: no
def create_img_8bit(path, period=4, grid=4, size=256):
    periodic = Periodic(period, grid, size)
    arr = periodic.noise2()

    arr *= 255
    arr = arr.astype(np.uint8)
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
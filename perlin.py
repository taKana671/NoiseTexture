import random

import cv2
import numpy as np

from noise import Noise


class Perlin:

    def __init__(self, grid=4, size=256):
        super().__init__()
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

    def pnoise2(self, x, y):
        p = np.array([x, y])
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable2(n + (arr := np.array([i, j])), f - arr)
             for j in range(2) for i in range(2)]

        f = 6 * f**5 - 15 * f**4 + 10 * f**3
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def pnoise3(self, x, y, t):
        p = np.array([x, y, t])
        n = np.floor(p)
        f, _ = np.modf(p)

        # 0.70710678 = 1 / sqrt(2)
        v = [self.gtable3(n + (arr := np.array([i, j, k])), f - arr) * 0.70710678
             for k in range(2) for j in range(2) for i in range(2)]

        f = 6 * f**5 - 15 * f**4 + 10 * f**3
        w0 = self.mix(self.mix(v[0], v[1], f[0]), self.mix(v[2], v[3], f[0]), f[1])
        w1 = self.mix(self.mix(v[4], v[5], f[0]), self.mix(v[6], v[7], f[0]), f[1])
        return 0.5 * self.mix(w0, w1, f[2]) + 0.5

    def noise2(self, t=None):
        if t is None:
            t = random.uniform(0, 1000)

        self.hash = {}

        arr = np.array(
            [self.pnoise2(x + t, y + t) for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def noise3(self, t=None):
        if t is None:
            t = random.uniform(0, 1000)

        self.hash = {}

        arr = np.array(
            [self.pnoise3(x + t, y + t, t) for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr


# np.count_nonzero(np.sign(arr) < 0) ; no less than zero: no
def create_img_8bit(path, grid=4, size=256):
    perlin = Perlin(grid, size)
    arr = perlin.noise3()
    # arr = perlin.noise2()

    # arr = np.abs(arr)
    arr *= 255
    arr = arr.astype(np.uint8)
    cv2.imwrite(path, arr)


def create_img_16bit(path, grid=4, size=256):
    perlin = Perlin(grid, size)
    arr = perlin.noise3()

    # img = np.abs(arr)
    arr *= 65535
    arr = arr.astype(np.uint16)
    cv2.imwrite(path, arr)


if __name__ == '__main__':
<<<<<<< HEAD
    # create_img_8bit('perlin_sample04.png')
    create_img_16bit('perlin_sample01.png')
=======
    perlin = Perlin(size=257)
    arr = perlin.create_noise(4)
    create_8_bit_img(arr, 'noise_sample.png')
>>>>>>> 63c59a3cddc55c7152b87ff82013e3e6f99edfca

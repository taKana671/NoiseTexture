import numpy as np

from noise import Noise


class Perlin(Noise):

    def __init__(self, weight=0.5, grid=4, size=256):
        super().__init__()
        self.size = size
        self.grid = grid
        self.weight = weight

    def pnoise2(self, p):
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable2(n + (arr := np.array([i, j])), f - arr)
             for j in range(2) for i in range(2)]

        f = 6 * f**5 - 15 * f**4 + 10 * f**3
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def pnoise3(self, p):
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
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.pnoise2(np.array([x + t, y + t]))
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def noise3(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.pnoise3(np.array([x + t, y + t, t]))
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def wrap2(self, x, y, rot=False):
        v = 0.0

        for i in range(4):
            cx = np.cos(2 * np.pi * v) if rot else v
            sy = np.sin(2 * np.pi * v) if rot else v
            _x = x + self.weight * cx
            _y = y + self.weight * sy
            v = self.pnoise2(np.array([_x, _y]))

        return v
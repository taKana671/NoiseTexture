import numpy as np
from datetime import datetime

from pynoise.noise import Noise
import cv2


class Simplex:

    def __init__(self, weight=0.5, grid=4, size=256):
        # super().__init__(grid, size)
        self.weight = weight
        self.size = 256
        self.grid = grid

    def mod_289(self, p):
        return p - np.floor(p * (1.0 / 289.0)) * 289

    def permute(self, p):
        return self.mod_289(((p * 34.0) + 1.0) * p)

    def fract(self, p):
        return p - np.floor(p)

    def inverssqrt(self, p):
        # return 1 / p ** 2
        return 1.79284291400159 - 0.85373472095314 * (p)

    def snoise(self, p):
        # skewed triangular grid
        grid = np.array([
            x := (3.0 - 3.0 ** 0.5) / 6.0,
            0.5 * (3.0 ** 0.5 - 1.0),
            -1.0 + 2.0 * x,
            1.0 / 41.0
        ])

        # the first corner
        i0 = np.floor(p + np.dot(p, np.repeat(grid[1], 2)))
        x0 = p - i0 + np.dot(i0, np.repeat(grid[0], 2))

        # the other two corners
        i1 = np.array([1.0, 0.0]) if x0[0] > x0[1] else np.array([0.0, 1.0])
        x1 = x0 + np.repeat(grid[0], 2) - i1
        x2 = x0 + np.repeat(grid[2], 2)

        i0 = self.mod_289(i0)
        perm = self.permute(i0[1] + np.array([0.0, i1[1], 1.0]))
        perm = self.permute(perm + i0[0] + np.array([0.0, i1[0], 1.0]))

        v = np.array([np.dot(x0, x0), np.dot(x1, x1), np.dot(x2, x2)])
        m = np.maximum(0.5 - v, 0)
        m = m ** 4

        x = 2.0 * self.fract(perm * np.repeat(grid[3], 3)) - 1.0
        h = np.abs(x) - 0.5
        ox = np.floor(x + 0.5)
        a0 = x - ox
        m *= self.inverssqrt(a0 ** 2 + h ** 2)

        g = np.array([
            a0[0] * x0[0] + h[0] * x0[1],
            a0[1] * x1[0] + h[1] * x1[1],
            a0[2] * x2[0] + h[2] * x2[1]
        ])

        return 130.0 * np.dot(m, g) * 0.5 + 0.5

    def noise(self):
        # arr = np.array(
        #     [self.snoise(np.array([x, y]) * 10)
        #         for y in np.linspace(0, self.grid, self.size)
        #         for x in np.linspace(0, self.grid, self.size)]
        # )
        arr = np.array(
            [self.snoise(np.array([x / 256, y / 256]) * 20)
                for y in range(256)
                for x in range(256)]
        )

        arr = arr.reshape(self.size, self.size)
        arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
        cv2.imwrite('simplex.png', arr)
        return arr







    # def gtable2(self, lattice, p):
    #     lattice = lattice.astype(np.uint32)

    #     if (tup := tuple(lattice)) in self.hash:
    #         idx = self.hash[tup]
    #     else:
    #         self.uhash22(lattice)
    #         idx = lattice[0] >> 29
    #         self.hash[tup] = idx

    #     u = (p[0] if idx < 4 else p[1]) * 0.92387953   # 0.92387953 = cos(pi/8)
    #     v = (p[1] if idx < 4 else p[0]) * 0.38268343   # 0.38268343 = sin(pi/8)

    #     _u = u if idx & 1 == 0 else -u
    #     _v = v if idx & 2 == 0 else -v

    #     return _u + _v

    # def gtable3(self, lattice, p):
    #     lattice = lattice.astype(np.uint32)

    #     if (tup := tuple(lattice)) in self.hash:
    #         idx = self.hash[tup]
    #     else:
    #         self.uhash33(lattice)
    #         idx = lattice[0] >> 28
    #         self.hash[tup] = idx

    #     u = p[0] if idx < 8 else p[1]

    #     if idx < 4:
    #         v = p[1]
    #     elif idx == 12 or idx == 14:
    #         v = p[0]
    #     else:
    #         v = p[2]

    #     _u = u if idx & 1 == 0 else -u
    #     _v = v if idx & 2 == 0 else -v

    #     return _u + _v

    # def pnoise2(self, p):
    #     n = np.floor(p)
    #     f, _ = np.modf(p)

    #     v = [self.gtable2(n + (arr := np.array([i, j])), f - arr)
    #          for j in range(2) for i in range(2)]

    #     f = self.fade(f)
    #     w0 = self.mix(v[0], v[1], f[0])
    #     w1 = self.mix(v[2], v[3], f[0])
    #     return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    # def pnoise3(self, p):
    #     n = np.floor(p)
    #     f, _ = np.modf(p)

    #     # 0.70710678 = 1 / sqrt(2)
    #     v = [self.gtable3(n + (arr := np.array([i, j, k])), f - arr) * 0.70710678
    #          for k in range(2) for j in range(2) for i in range(2)]

    #     f = self.fade(f)
    #     w0 = self.mix(self.mix(v[0], v[1], f[0]), self.mix(v[2], v[3], f[0]), f[1])
    #     w1 = self.mix(self.mix(v[4], v[5], f[0]), self.mix(v[6], v[7], f[0]), f[1])
    #     return 0.5 * self.mix(w0, w1, f[2]) + 0.5

    # def noise2(self, t=None):
    #     t = self.mock_time() if t is None else t
    #     self.hash = {}

    #     arr = np.array(
    #         [self.pnoise2(np.array([x + t, y + t]))
    #             for y in np.linspace(0, self.grid, self.size)
    #             for x in np.linspace(0, self.grid, self.size)]
    #     )
    #     arr = arr.reshape(self.size, self.size)
    #     return arr

    # def noise3(self, t=None):
    #     t = self.mock_time() if t is None else t
    #     self.hash = {}

    #     arr = np.array(
    #         [self.pnoise3(np.array([x + t, y + t, t]))
    #             for y in np.linspace(0, self.grid, self.size)
    #             for x in np.linspace(0, self.grid, self.size)]
    #     )
    #     arr = arr.reshape(self.size, self.size)
    #     return arr

    # def wrap2(self, x, y, rot=False):
    #     v = 0.0

    #     for _ in range(4):
    #         cx = np.cos(2 * np.pi * v) if rot else v
    #         sy = np.sin(2 * np.pi * v) if rot else v
    #         _x = x + self.weight * cx
    #         _y = y + self.weight * sy
    #         v = self.pnoise2(np.array([_x, _y]))

    #     return v
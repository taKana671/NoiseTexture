import cv2
import numpy as np

from pynoise.noise import Noise


class Simplex(Noise):

    def __init__(self, width=256, height=256, space_scale=20):
        # self.size = 256
        self.space_scale = space_scale
        self.width = width
        self.height = height

    def mod289(self, p):
        return p - np.floor(p * (1.0 / 289.0)) * 289.0

    def permute(self, p):
        return self.mod289(((p * 34.0) + 1.0) * p)

    def inverssqrt(self, p):
        """Normalise gradients implicitly by scaling m
           Approximation of: m *= inversesqrt( a0*a0 + h*h )
        """
        return 1.79284291400159 - 0.85373472095314 * p
        # return 1 / p ** 2

    def snoise2(self, p):
        # skewed triangular grid
        grid = [
            x := (3.0 - 3.0 ** 0.5) / 6.0,
            0.5 * (3.0 ** 0.5 - 1.0),
            -1.0 + 2.0 * x,
            1.0 / 41.0
        ]

        # the first corner
        i = np.floor(p + np.dot(p, np.repeat(grid[1], 2)))
        x0 = p - i + np.dot(i, np.repeat(grid[0], 2))

        # the other two corners
        i1 = np.array([1.0, 0.0]) if x0[0] > x0[1] else np.array([0.0, 1.0])
        x1 = x0 + np.repeat(grid[0], 2) - i1
        x2 = x0 + np.repeat(grid[2], 2)

        # Do some permutations to avoid truncation effects in permutation
        i = self.mod289(i)
        perm = self.permute(i[1] + np.array([0.0, i1[1], 1.0]))
        perm = self.permute(perm + i[0] + np.array([0.0, i1[0], 1.0]))

        v = np.array([np.dot(x0, x0), np.dot(x1, x1), np.dot(x2, x2)])
        m = np.maximum(0.5 - v, 0.0)
        m = m ** 4

        # Gradients: 41 pts uniformly over a line, mapped onto a diamond
        # The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
        x = 2.0 * self.fract(perm * np.repeat(grid[3], 3)) - 1.0
        h = np.abs(x) - 0.5
        ox = np.floor(x + 0.5)
        a0 = x - ox
        m *= self.inverssqrt(a0 ** 2 + h ** 2)

        # Compute final noise value
        g = np.array([
            a0[0] * x0[0] + h[0] * x0[1],
            a0[1] * x1[0] + h[1] * x1[1],
            a0[2] * x2[0] + h[2] * x2[1]
        ])

        return 130.0 * np.dot(m, g) * 0.5 + 0.5

    def snoise3(self, p):
        c = np.array([1.0 / 6.0, 1.0 / 3.0])
        d = np.array([0.0, 0.5, 1.0, 2.0])

        # the first corner
        i = np.floor(p + np.dot(p, np.repeat(c[1], 3)))
        x0 = p - i + np.dot(i, np.repeat(c[0], 3))

        # other corners
        g = np.array([self.step(v1, v2) for v1, v2 in zip(x0[[1, 2, 0]], x0)])
        ll = 1.0 - g
        i1 = np.minimum(g, ll[[2, 0, 1]])
        i2 = np.maximum(g, ll[[2, 0, 1]])

        x1 = x0 - i1 + np.repeat(c[0], 3)
        x2 = x0 - i2 + np.repeat(c[1], 3)
        x3 = x0 - np.repeat(d[1], 3)

        # Do some permutations to avoid truncation effects in permutation
        i = self.mod289(i)
        perm = self.permute(i[2] + np.array([0.0, i1[2], i2[2], 1.0]))
        perm = self.permute(perm + i[1] + np.array([0.0, i1[1], i2[1], 1.0]))
        perm = self.permute(perm + i[0] + np.array([0.0, i1[0], i2[0], 1.0]))

        # Gradients: 7x7 points over a square, mapped onto an octahedron.
        # The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
        ns = (1.0 / 7.0) * d[[3, 1, 2]] - d[[0, 2, 0]]
        j = perm - 49.0 * np.floor(perm * ns[2] * ns[2])

        _x = np.floor(j * ns[2])
        _y = np.floor(j - 7.0 * _x)
        x = _x * ns[0] + np.repeat(ns[1], 4)
        y = _y * ns[0] + np.repeat(ns[1], 4)
        h = 1.0 - np.abs(x) - np.abs(y)

        b0 = np.concatenate([x[:2], y[:2]])
        b1 = np.concatenate([x[2:], y[2:]])
        s0 = np.floor(b0) * 2.0 + 1.0
        s1 = np.floor(b1) * 2.0 + 1.0
        sh = np.array([self.step(v, 0) for v in h]) * -1
        a0 = b0[[0, 2, 1, 3]] + s0[[0, 2, 1, 3]] * sh[[0, 0, 1, 1]]
        a1 = b1[[0, 2, 1, 3]] + s1[[0, 2, 1, 3]] * sh[[2, 2, 3, 3]]

        p0 = np.concatenate([a0[:2], h[:1]])
        p1 = np.concatenate([a0[2:], h[1:2]])
        p2 = np.concatenate([a1[:2], h[2:3]])
        p3 = np.concatenate([a1[2:], h[3:]])

        # normalize gradients
        nm = self.inverssqrt(
            np.array([np.dot(p0, p0), np.dot(p1, p1), np.dot(p2, p2), np.dot(p3, p3)]))
        p0 *= nm[0]
        p1 *= nm[1]
        p2 *= nm[2]
        p3 *= nm[3]

        arr = np.array([np.dot(x0, x0), np.dot(x1, x1), np.dot(x2, x2), np.dot(x3, x3)])
        m = np.maximum(0.6 - arr, 0)
        m = m ** 4

        arr = np.array([np.dot(p0, x0), np.dot(p1, x1), np.dot(p2, x2), np.dot(p3, x3)])
        return 42.0 * np.dot(m, arr) * 0.5 + 0.5

    def noise2(self, t=None):
        t = self.mock_time() if t is None else t
        m = min(self.width, self.height)

        arr = np.array(
            [self.snoise2(np.array([x, y]) + t / m * self.space_scale)
                for y in range(self.height)
                for x in range(self.width)]
        )

        arr = arr.reshape(self.height, self.width)
        # arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
        # cv2.imwrite('simplex.png', arr)
        return arr

    def noise3(self, t=None):
        t = self.mock_time() if t is None else t
        m = min(self.width, self.height)

        arr = np.array(
            [self.snoise3(np.array([x, y, t]) / m * self.space_scale)
                for y in range(self.height)
                for x in range(self.width)]
        )

        arr = arr.reshape(self.height, self.width)
        # arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
        # cv2.imwrite('simplex.png', arr)
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
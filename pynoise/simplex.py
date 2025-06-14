# import cv2
import numpy as np
from functools import reduce

from .noise import Noise
from .fBm import Fractal2D, Fractal3D


class SimplexNoise(Noise):

    def mod289(self, p):
        return p - np.floor(p * (1.0 / 289.0)) * 289.0

    def permute(self, p):
        return self.mod289(((p * 34.0) + 1.0) * p)

    def inverssqrt(self, p):
        return 1.79284291400159 - 0.85373472095314 * p
        # return 1 / p ** 2

    def snoise2(self, x, y):
        # skewed triangular grid
        grid = [
            v := (3.0 - 3.0 ** 0.5) / 6.0,
            0.5 * (3.0 ** 0.5 - 1.0),
            -1.0 + 2.0 * v,
            1.0 / 41.0
        ]

        # the first corner
        p = np.array([x, y])
        inner_prod = p[0] * grid[1] + p[1] * grid[1]
        d0 = np.floor(p + inner_prod)

        inner_prod = d0[0] * grid[0] + d0[1] * grid[0]
        x0 = p - d0 + inner_prod

        # the other two corners
        d1 = np.array([1.0, 0.0]) if x0[0] > x0[1] else np.array([0.0, 1.0])
        x1 = x0 + grid[0] - d1
        x2 = x0 + grid[2]

        # Do some permutations to avoid truncation effects in permutation
        d0 = self.mod289(d0)
        perm = np.zeros(3)

        for i in range(1, -1, -1):
            v = d0[i]
            perm[0] = self.permute(perm[0] + v)
            perm[1] = self.permute(perm[1] + v + d1[i])
            perm[2] = self.permute(perm[2] + v + 1.0)

        inner_prods = [np.dot(c, c) for c in (x0, x1, x2)]
        m = np.array([max(0.5 - v, 0.0) ** 4 for v in inner_prods])

        # Gradients: 41 pts uniformly over a line, mapped onto a diamond
        # The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
        f, _ = np.modf(perm * grid[3])
        fx = 2.0 * f - 1.0
        h = np.abs(fx) - 0.5
        ox = np.floor(fx + 0.5)
        a0 = fx - ox
        # Normalise gradients implicitly by scaling m
        # Approximation of: m *= inversesqrt( a0*a0 + h*h)
        m *= self.inverssqrt(a0 ** 2 + h ** 2)

        # Compute final noise value
        g = [
            a0[0] * x0[0] + h[0] * x0[1],
            a0[1] * x1[0] + h[1] * x1[1],
            a0[2] * x2[0] + h[2] * x2[1]
        ]

        return 130.0 * np.dot(m, g) * 0.5 + 0.5

    def snoise3(self, x, y, z):
        c = [1.0 / 6.0, 1.0 / 3.0]
        q = [0.0, 0.5, 1.0, 2.0]

        # the first corner
        p = np.array([x, y, z])
        inner_prod = reduce(lambda ret, v: ret + v * c[1], p, 0)
        d0 = np.floor(p + inner_prod)

        inner_prod = reduce(lambda ret, v: ret + v * c[0], d0, 0)
        x0 = p - d0 + inner_prod

        # other corners
        g = [self.step(v1, v2) for v1, v2 in zip(x0[[1, 2, 0]], x0)]
        ll = [1.0 - v for v in g]
        ll = [ll[2], ll[0], ll[1]]
        d1 = np.minimum(g, ll)
        d2 = np.maximum(g, ll)

        x1 = x0 - d1 + c[0]
        x2 = x0 - d2 + c[1]
        x3 = x0 - q[1]

        # Do some permutations to avoid truncation effects in permutation
        d0 = self.mod289(d0)
        perm = np.zeros(4)

        for i in range(2, -1, -1):
            m = d0[i]
            perm[0] = self.permute(perm[0] + m)
            perm[1] = self.permute(perm[1] + m + d1[i])
            perm[2] = self.permute(perm[2] + m + d2[i])
            perm[3] = self.permute(perm[3] + m + 1.0)

        # Gradients: 7x7 points over a square, mapped onto an octahedron.
        # The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
        v = 1.0 / 7.0
        ns = [v * q[i0] - q[i1] for i0, i1 in zip([3, 1, 2], [0, 2, 0])]
        j = perm - 49.0 * np.floor(perm * ns[2] * ns[2])

        _xx = np.floor(j * ns[2])
        _yy = np.floor(j - 7.0 * _xx)

        xx = _xx * ns[0] + ns[1]
        yy = _yy * ns[0] + ns[1]
        h = 1.0 - np.abs(xx) - np.abs(yy)

        b0 = np.concatenate([xx[:2], yy[:2]])
        b1 = np.concatenate([xx[2:], yy[2:]])
        s0 = np.floor(b0) * 2.0 + 1.0
        s1 = np.floor(b1) * 2.0 + 1.0
        sh = np.array([-self.step(v, 0.0) for v in h])

        a0 = b0[[0, 2, 1, 3]] + s0[[0, 2, 1, 3]] * sh[[0, 0, 1, 1]]
        a1 = b1[[0, 2, 1, 3]] + s1[[0, 2, 1, 3]] * sh[[2, 2, 3, 3]]

        p0 = np.concatenate([a0[:2], h[:1]])
        p1 = np.concatenate([a0[2:], h[1:2]])
        p2 = np.concatenate([a1[:2], h[2:3]])
        p3 = np.concatenate([a1[2:], h[3:]])

        p0 *= self.inverssqrt(np.dot(p0, p0))
        p1 *= self.inverssqrt(np.dot(p1, p1))
        p2 *= self.inverssqrt(np.dot(p2, p2))
        p3 *= self.inverssqrt(np.dot(p3, p3))

        pts = (p0, p1, p2, p3)
        cns = (x0, x1, x2, x3)
        mx = [max(0.6 - np.dot(co, co), 0) ** 4 for co in cns]
        cp = [np.dot(pt, co) for pt, co in zip(pts, cns)]

        return 42.0 * np.dot(mx, cp) * 0.5 + 0.5

    def noise2(self, width=256, height=256, scale=20, t=None):
        t = self.mock_time() if t is None else t
        m = min(width, height)

        arr = np.array(
            [self.snoise2((x / m + t) * scale, (y / m + t) * scale)
                for y in range(height)
                for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr

    def noise3(self, width=256, height=256, scale=20, t=None):
        t = self.mock_time() if t is None else t
        m = min(width, height)

        arr = np.array(
            [self.snoise3((x / m + t) * scale, (y / m + t) * scale, t * scale)
                for y in range(height)
                for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr

    def fractal2(self, width=256, height=256, t=None,
                 gain=0.5, lacunarity=2.01, octaves=4):
        t = self.mock_time() if t is None else t
        m = min(width, height)

        noise = Fractal2D(self.snoise2)

        arr = np.array(
            [noise.fractal(x / m + t, y / m + t)
             for y in range(height) for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr

    def fractal3(self, width=256, height=256, t=None,
                 gain=0.5, lacunarity=2.01, octaves=4):
        t = self.mock_time() if t is None else t
        m = min(width, height)
        noise = Fractal3D(self.snoise3)

        arr = np.array(
            [noise.fractal(x / m + t, y / m + t, t)
             for y in range(height) for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr
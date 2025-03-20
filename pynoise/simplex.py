import cv2
import numpy as np

from pynoise.noise import Noise


class Simplex(Noise):

    # def __init__(self, weight=0.5, lacunarity=2.01, octaves=4,
    #              width=256, height=256, space_scale=20):
    # def __init__(self, width=256, height=256, space_scale=20):
    #     # self.weight = weight
    #     # self.lacunarity = lacunarity
    #     # self.octaves = octaves
    #     self.scale = space_scale
    #     self.width = width
    #     self.height = height

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
        perm = 0

        for idx in range(1, -1, -1):
            perm = self.permute(perm + i[idx] + np.array([0.0, i1[idx], 1.0]))

        # perm = self.permute(i[1] + np.array([0.0, i1[1], 1.0]))
        # perm = self.permute(perm + i[0] + np.array([0.0, i1[0], 1.0]))

        v = np.array([np.dot(c, c) for c in (x0, x1, x2)])
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
        g = self.step_arr(x0[[1, 2, 0]], x0)
        ll = 1.0 - g
        i1 = np.minimum(g, ll[[2, 0, 1]])
        i2 = np.maximum(g, ll[[2, 0, 1]])

        x1 = x0 - i1 + np.repeat(c[0], 3)
        x2 = x0 - i2 + np.repeat(c[1], 3)
        x3 = x0 - np.repeat(d[1], 3)

        # Do some permutations to avoid truncation effects in permutation
        i = self.mod289(i)
        perm = 0

        for idx in range(2, -1, -1):
            perm = self.permute(perm + i[idx] + np.array([0.0, i1[idx], i2[idx], 1.0]))

        # perm = self.permute(i[2] + np.array([0.0, i1[2], i2[2], 1.0]))
        # perm = self.permute(perm + i[1] + np.array([0.0, i1[1], i2[1], 1.0]))
        # perm = self.permute(perm + i[0] + np.array([0.0, i1[0], i2[0], 1.0]))

        # print('perm', perm)
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
        sh = self.step_arr(h, [0] * 4) * -1
        a0 = b0[[0, 2, 1, 3]] + s0[[0, 2, 1, 3]] * sh[[0, 0, 1, 1]]
        a1 = b1[[0, 2, 1, 3]] + s1[[0, 2, 1, 3]] * sh[[2, 2, 3, 3]]

        p0 = np.concatenate([a0[:2], h[:1]])
        p1 = np.concatenate([a0[2:], h[1:2]])
        p2 = np.concatenate([a1[:2], h[2:3]])
        p3 = np.concatenate([a1[2:], h[3:]])

        # normalize gradients
        pts = (p0, p1, p2, p3)
        corners = (x0, x1, x2, x3)

        nm = self.inverssqrt(np.array([np.dot(pt, pt) for pt in pts]))
        p0 *= nm[0]
        p1 *= nm[1]
        p2 *= nm[2]
        p3 *= nm[3]

        arr = np.array([np.dot(co, co) for co in corners])
        m = np.maximum(0.6 - arr, 0)
        # m = m ** 4

        arr = np.array([np.dot(pt, co) for pt, co in zip(pts, corners)])
        # return 42.0 * np.dot(m, arr) * 0.5 + 0.5
        return 42.0 * np.dot(m ** 4, arr) * 0.5 + 0.5

    # def fractal(self, noise, p):
    #     """Args:
    #         noise (callable): a function to generate noise.
    #         p (Numpy.ndarary): point
    #     """
    #     v = 0.0
    #     amp = 1.0
    #     freq = 1.0

    #     for _ in range(self.octaves):
    #         v += amp * (noise(freq * p) - 0.5)
    #         amp *= self.weight
    #         freq *= self.lacunarity

    #     return 0.5 * v + 0.5

    def noise2(self, width=256, height=256, scale=20, t=None):
        t = self.mock_time() if t is None else t
        m = min(self.width, self.height)

        # arr = np.array(
        #     [self.snoise2((np.array([x + t, y + t]) / m) * self.scale)
        #         for y in range(self.height)
        #         for x in range(self.width)]
        # )

        arr = np.array(
            [self.snoise2((np.array([x, y]) / m + t) * scale)
                for y in range(height)
                for x in range(width)]
        )

        arr = arr.reshape(self.height, self.width)
        # arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
        # cv2.imwrite('simplex.png', arr)
        return arr

    def noise3(self, width=256, height=256, scale=20, t=None):
        t = self.mock_time() if t is None else t
        m = min(self.width, self.height)

        # arr = np.array(
        #     [self.snoise3((np.array([x / m, y / m, t])) * self.scale)
        #         for y in range(self.height)
        #         for x in range(self.width)]
        # )
        arr = np.array(
            [self.snoise3((np.array([x / m + t, y / m + t, t])) * scale)
                for y in range(height)
                for x in range(width)]
        )

        arr = arr.reshape(self.height, self.width)
        # arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
        # cv2.imwrite('simplex.png', arr)
        return arr

    # def noise3_fractal(self, t=None):
    #     t = self.mock_time() if t is None else t
    #     m = min(self.width, self.height)

    #     arr = np.array(
    #         [self.fractal((np.array([x / m, y / m, t])) * self.scale)
    #             for y in range(self.height)
    #             for x in range(self.width)]
    #     )

    #     arr = arr.reshape(self.height, self.width)
    #     # arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
    #     # cv2.imwrite('simplex.png', arr)
    #     return arr
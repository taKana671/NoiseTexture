import random

import numpy as np


k = np.array([1164413355, 1737075525, 2309703015], dtype=np.uint32)
u = np.array([1, 2, 3], dtype=np.uint32)
UINT_MAX = np.iinfo(np.uint32).max


class Noise:

    def __init__(self, grid, size):
        self.size = size
        self.grid = grid

    def mock_time(self):
        return random.uniform(0, 1000)

    def uhash11(self, n):
        n ^= n << u[0]
        n ^= n >> u[0]
        n *= k[0]
        n ^= n << u[0]
        n *= k[0]

    def uhash22(self, n):
        n ^= n[::-1] << u[:2]
        n ^= n[::-1] >> u[:2]
        n *= k[:2]
        n ^= n[::-1] << u[:2]
        n *= k[:2]

    def uhash33(self, n):
        n ^= n[[1, 2, 0]] << u
        n ^= n[[1, 2, 0]] >> u
        n *= k
        n ^= n[[1, 2, 0]] << u
        n *= k

    def hash21(self, p):
        n = p.astype(np.uint32)

        if (key := tuple(n)) in self.hash:
            return self.hash[key]

        self.uhash22(n)
        h = n[0] / UINT_MAX
        self.hash[key] = h
        return h

    def hash22(self, p):
        n = p.astype(np.uint32)

        if (key := tuple(n)) in self.hash:
            return self.hash[key]

        self.uhash22(n)
        h = n / UINT_MAX
        self.hash[key] = h
        return h

    def hash33(self, p):
        n = p.astype(np.uint32)

        if (key := tuple(n)) in self.hash:
            return self.hash[key]

        self.uhash33(n)
        h = n / UINT_MAX
        self.hash[key] = h
        return h

    def mix(self, x, y, a):
        return x + (y - x) * a

    def step(self, a, x):
        if x <= a:
            return 0
        return 1

    def smoothstep(self, edge0, edge1, x):
        """Args:
            edge0, edge1, x (float)
        """
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def xy2pol(self, x, y):
        r = (x ** 2 + y ** 2) ** 0.5

        if x == 0:
            x = np.sign(y) * np.pi / 2
        else:
            x = np.arctan2(y, x)

        return x, r

    def get_norm(self, vec):
        return sum(v ** 2 for v in vec) ** 0.5

    def fade(self, x):
        return 6 * x**5 - 15 * x**4 + 10 * x**3

    def wrap(self, t=None, rot=False):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.wrap2(x + t, y + t, rot)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr
import cmath
import random

import numpy as np

k = ('0x456789ab', '0x6789ab45', '0x89ab4567')
k = np.array([int(v, 0) for v in k], dtype=np.uint)
u = np.array([1, 2, 3], dtype=np.uint)
# k = np.array([int(v, 0) for v in k])
# u = np.array([1, 2, 3])

# UINT_MAX = int('0xffffffff', 0)
UINT_MAX = np.iinfo(np.uint).max



class Noise:

    def __init__(self):
        self.hash = {}

    def uhash11(self, n):
        n ^= n << u[0]
        n ^= n >> u[0]
        n *= k[0]
        n ^= n << u[0]
        return n * k[0]

    def uhash22(self, n):
        n ^= n[::-1] << u[:2]
        n ^= n[::-1] >> u[:2]
        n *= k[:2]
        n ^= n[::-1] << u[:2]
        return n * k[:2]

    # kernprof -l -v cellular.py

    # @profile
    def uhash33(self, n):
        # print(n)
        # *************************************
        n ^= np.array([n[1], n[2], n[0]]) << u
        n ^= np.array([n[1], n[2], n[0]]) >> u
        n *= k
        n ^= np.array([n[1], n[2], n[0]]) << u
        return n * k

        # *************************************
        # n ^= n[[1, 2, 0]] << u
        # n ^= n[[1, 2, 0]] >> u
        # n *= k
        # n ^= n[[1, 2, 0]] << u
        # return n * k

    def hash21(self, p):
        n = p.astype(np.uint)

        if (key := tuple(n)) in self.hash:
            h = self.hash[key]
        else:
            h = self.uhash22(n)[0]
            self.hash[key] = h

        return h / UINT_MAX
        # return self.uhash22(n)[0] / UINT_MAX

    def hash22(self, p):
        n = p.astype(np.uint)

        if (key := tuple(n)) in self.hash:
            h = self.hash[key]
        else:
            h = self.uhash22(n)
            self.hash[key] = h

        return h / UINT_MAX

    def hash33(self, p):
        n = p.astype(np.uint)

        if (key := tuple(n)) in self.hash:
            h = self.hash[key]
        else:
            h = self.uhash33(n)
            self.hash[key] = h

        return h / UINT_MAX

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

    def convert(self, v, n):
        return (np.floor(n * v) + self.step(0.5, np.modf(n * v)[0])) / n

    def wrap(self, t=None, rot=False):
        if t is None:
            t = random.uniform(0, 1000)

        self.hash = {}

        arr = np.array(
            [self.wrap2(x + t, y + t, rot) for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def convert_gradation(self, t=None, rot=False):
        if t is None:
            t = random.uniform(0, 1000)

        self.hash = {}

        arr = np.array(
            [self.convert(self.wrap2(x + t, y + t, rot), t) for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

        

        # # p = np.arctan2(y, x)
        # r, p = cmath.polar(complex(x, y)) 
        # return p, y
        # cmath.polar(complex(x, y)); r, p

    # def pol2xy(self, r, p):
    #     x = r * np.cos(p)
    #     y = r * np.sin(p)
    #     return x, y
    #     # z = r * cmath.exp(1j * p)
    #     # return z.real, z.imag
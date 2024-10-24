# cython: profile=True
# cython: language_level=3

import random

import numpy as np
cimport numpy as cnp


# k = np.array([1164413355, 1737075525, 2309703015], dtype=np.uint)
# u = np.array([1, 2, 3], dtype=np.uint)
UINT_MAX = np.iinfo(np.uint).max


cdef class Noise:

    def __init__(self):
        self.k = [1164413355, 1737075525, 2309703015]
        self.u = [1, 2, 3]

    def mock_time(self):
        return random.uniform(0, 1000)

    def uhash11(self, n):
        k = np.array([1164413355, 1737075525, 2309703015], dtype=np.uint)
        u = np.array([1, 2, 3], dtype=np.uint)

        n ^= n << u[0]
        n ^= n >> u[0]
        n *= k[0]
        n ^= n << u[0]
        return n * k[0]

    def uhash22(self, n):
        k = np.array([1164413355, 1737075525, 2309703015], dtype=np.uint)
        u = np.array([1, 2, 3], dtype=np.uint)

        n ^= n[::-1] << u[:2]
        n ^= n[::-1] >> u[:2]
        n *= k[:2]
        n ^= n[::-1] << u[:2]
        return n * k[:2]

    cdef void uhash33(self, unsigned int[3] *n):

        # cdef:
        #     unsigned int[3] k = [1164413355, 1737075525, 2309703015]
        #     unsigned int[3] u = [1, 2, 3]

        n[0][0] ^= n[0][1] << self.u[0]
        n[0][1] ^= n[0][2] << self.u[1]
        n[0][2] ^= n[0][0] << self.u[2]

        n[0][0] ^= n[0][1] >> self.u[0]
        n[0][1] ^= n[0][2] >> self.u[1]
        n[0][2] ^= n[0][0] >> self.u[2]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]
        n[0][2] *= self.k[2]

        n[0][0] ^= n[0][1] << self.u[0]
        n[0][1] ^= n[0][2] << self.u[1]
        n[0][2] ^= n[0][0] << self.u[2] 

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]
        n[0][2] *= self.k[2]

    def hash21(self, p):
        n = p.astype(np.uint)
        h = self.uhash22(n)[0]
        # if (key := tuple(n)) in self.hash:
        #     h = self.hash[key]
        # else:
        #     h = self.uhash22(n)[0]
        #     self.hash[key] = h

        return h / UINT_MAX

    def hash22(self, p):
        n = p.astype(np.uint)
        h = self.uhash22(n)

        # if (key := tuple(n)) in self.hash:
        #     h = self.hash[key]
        # else:
        #     h = self.uhash22(n)
        #     self.hash[key] = h

        return h / UINT_MAX

    # def hash33(self, p):
        # n = p.astype(np.uint)
        # h = self.uhash33(n)

        # if (key := tuple(n)) in self.hash:
        #     h = self.hash[key]
        # else:
        #     h = self.uhash33(n)
        #     self.hash[key] = h

        # return h / UINT_MAX

    def gtable2(self, lattice, p):
        lattice = lattice.astype(np.uint)
        idx = self.uhash22(lattice)[0] >> 29

        # if (tup := tuple(lattice)) in self.hash:
        #     idx = self.hash[tup]
        # else:
        #     idx = random.randint(1, 6)
        #     self.hash[tup] = idx

        u = (p[0] if idx < 4 else p[1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[1] if idx < 4 else p[0]) * 0.38268343   # 0.38268343 = sin(pi/8)

        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    cdef double gtable3(self, unsigned int[3] *lattice, double[3] *p):

        cdef:
            unsigned int idx
            double u, v, _u, _v


        # lattice = lattice.astype(np.uint)
        self.uhash33(lattice)
        idx = lattice[0][0] >> 28

        # if (tup := tuple(lattice)) in self.hash:
        #     idx = self.hash[tup]
        # else:
        #     idx = random.randint(0, 15)
        #     self.hash[tup] = idx

        u = p[0][0] if idx < 8 else p[0][1]

        if idx < 4:
            v = p[0][1]
        elif idx == 12 or idx == 14:
            v = p[0][0]
        else:
            v = p[0][2]

        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    cdef double fade(self, double x):
        return 6 * x**5 - 15 * x**4 + 10 * x**3
    
    
    cdef double mix(self, double x, double y, double a):
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
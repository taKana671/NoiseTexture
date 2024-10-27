# cython: profile=True
# cython: language_level=3

import random

import cython
import numpy as np
cimport numpy as cnp


# k = np.array([1164413355, 1737075525, 2309703015], dtype=np.uint)
# u = np.array([1, 2, 3], dtype=np.uint)
UINT_MAX = np.iinfo(np.uint).max


cdef class Noise:

    def __init__(self, grid, size):
        self.size = size
        self.grid = grid

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

    cdef void uhash22(self, unsigned int[2] *n):
        n[0][0] ^= n[0][1] << self.u[0]
        n[0][1] ^= n[0][0] << self.u[1]

        n[0][0] ^= n[0][1] >> self.u[0]
        n[0][1] ^= n[0][0] >> self.u[1]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]

        n[0][0] ^= n[0][1] << self.u[0]
        n[0][1] ^= n[0][0] << self.u[1]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]

    cdef void uhash33(self, unsigned int[3] *n):
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


    @cython.cdivision(True)
    cdef double hash21(self, double[2] *p):
        cdef:
            unsigned int uint_max = 4294967295
            unsigned int[2] n
            double h

        for i in range(2):
            n[i] = <unsigned int>p[0][i]

        self.uhash22(&n)
        h = <double>n[0] / uint_max
        return h

    @cython.cdivision(True)
    cdef void hash22(self, double[2] *p, double[2] *h):
        cdef:
            unsigned i
            unsigned int uint_max = 4294967295
            unsigned int[2] n

        for i in range(2):
            n[i] = <unsigned int>p[0][i]

        self.uhash22(&n)

        for i in range(2):
            h[0][i] = <double>n[i] / uint_max

    @cython.cdivision(True)
    cdef void hash33(self, double[3] *p, double[3] *h):
        cdef:
            unsigned i
            unsigned int uint_max = 4294967295
            unsigned int[3] n

        for i in range(3):
            n[i] = <unsigned int>p[0][i]

        self.uhash33(&n)

        for i in range(3):
            h[0][i] = <double>n[i] / uint_max

    cdef double gtable2(self, unsigned int[2] *lattice, double[2] *p):
        cdef:
            unsigned int idx
            double u, v, _u, _v

        self.uhash22(lattice)
        idx = lattice[0][0] >> 29

        u = (p[0][0] if idx < 4 else p[0][1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[0][1] if idx < 4 else p[0][0]) * 0.38268343   # 0.38268343 = sin(pi/8)
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    cdef double gtable3(self, unsigned int[3] *lattice, double[3] *p):
        cdef:
            unsigned int idx
            double u, v, _u, _v

        self.uhash33(lattice)
        idx = lattice[0][0] >> 28

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

    cdef unsigned int step(self, double a, double x):
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

    cdef double get_norm3(self, double[3] *v):
        return (v[0][0] ** 2 + v[0][1] ** 2 + v[0][2] ** 2) ** 0.5

    cdef double get_norm2(self, double[2] *v):
        return (v[0][0] ** 2 + v[0][1] ** 2) ** 0.5
        
    #def get_norm(self, vec):
        #return sum(v ** 2 for v in vec) ** 0.5

    cpdef wrap(self, bint rot=False):
        t = self.mock_time()

        arr = np.array(
            [self.wrap2(x + t, y + t, rot)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    cdef double wrap2(self, double x, double y, bint rot=False):
        pass
# cython: language_level=3

import random

import cython
import numpy as np
cimport numpy as cnp
from libc.math cimport atan2, cos, sin, pi


cdef class Noise:

    def __init__(self, grid, size):
        self.size = size
        self.grid = grid

        self.k = [1164413355, 1737075525, 2309703015]
        self.u = [1, 2, 3]

    def mock_time(self):
        return random.uniform(0, 1000)

    cdef void uhash11(self, unsigned int *n):
        n[0] ^= n[0] << self.u[0]
        n[0] ^= n[0] >> self.u[0]
        n[0] *= self.k[0]
        n[0] ^= n[0] << self.u[0]
        n[0] *= self.k[0]

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
            unsigned int i
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
            unsigned int uint_max = 4294967295
            unsigned int i
            unsigned int[2] n

        for i in range(2):
            n[i] = <unsigned int>p[0][i]

        self.uhash22(&n)

        for i in range(2):
            h[0][i] = <double>n[i] / uint_max

    @cython.cdivision(True)
    cdef void hash33(self, double[3] *p, double[3] *h):
        cdef:
            unsigned int uint_max = 4294967295
            unsigned int i
            unsigned int[3] n

        for i in range(3):
            n[i] = <unsigned int>p[0][i]

        self.uhash33(&n)

        for i in range(3):
            h[0][i] = <double>n[i] / uint_max

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

    @cython.cdivision(True)
    cdef double sign_with_abs(self, double *x):
        if abs(x[0]) == 0:
            return 0.0
   
        return x[0] / abs(x[0])
    
    @cython.cdivision(True)
    cdef (double, double) xy2pol(self, double x, double y):
        cdef:
            double r, px

        r = (x ** 2 + y ** 2) ** 0.5

        if x == 0:
            px = self.sign_with_abs(&y) * pi / 2
        else:
            px = atan2(y, x)

        return px, r

    cdef double get_norm3(self, double[3] *v):
        return (v[0][0] ** 2 + v[0][1] ** 2 + v[0][2] ** 2) ** 0.5

    cdef double get_norm2(self, double[2] *v):
        return (v[0][0] ** 2 + v[0][1] ** 2) ** 0.5

    cpdef wrap(self, rot=False, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self.wrap2(x + t, y + t, rot)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    cdef double wrap2(self, double x, double y, bint rot=False):
        pass
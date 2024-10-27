# cython: profile=True
# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport floor, cos, sin, pi

from noise cimport Noise


cdef class FractionalBrownianMotion(Noise):

    cdef:
        double weight

    def __init__(self, weight=0.5, grid=4, size=256):
        super().__init__(grid, size)
        self.weight = weight

    cdef double vnoise2(self, double x, double y):
        cdef:
            unsigned int i, j
            double[2] arr
            double fx, fy, ret, nx, ny
            double[4] v

        nx = floor(x)
        ny = floor(y)
        fx = x - nx
        fy = y - ny

        for j in range(2):
            for i in range(2):
                arr = [nx + i, ny + j]
                v[i + 2 * j] = self.hash21(&arr)

        fx = self.fade(fx)
        fy = self.fade(fy)

        w0 = self.mix(v[0], v[1], fx)
        w1 = self.mix(v[2], v[3], fx)
        return self.mix(w0, w1, fy)

    cdef double fbm2(self, double x, double y):
        cdef:
            double v = 0.0
            double amp = 1.0
            double freq = 1.0
            double freq_w = 2.011

        for _ in range(4):
            v += amp * (self.vnoise2(freq * x, freq * y) - 0.5)
            amp *= self.weight
            freq *= freq_w

        return 0.5 * v + 0.5

    cpdef noise2(self):
        t = self.mock_time()

        arr = np.array(
            [self.fbm2(x + t, y + t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )

        arr = arr.reshape(self.size, self.size)
        return arr

    cdef double wrap2(self, double x, double y, bint rot=False):
        cdef:
            double v = 0.0
            double cx, cy

        for i in range(4):
            cx = cos(2 * pi * v) if rot else v
            sy = sin(2 * pi * v) if rot else v
            x += self.weight * cx
            y += self.weight * sy
            v = self.fbm2(x, y)

        return v
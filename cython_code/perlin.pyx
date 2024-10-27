# cython: profile=True
# cython: language_level=3

import math
import numpy as np
cimport numpy as cnp
from libc.math cimport floor, cos, sin, pi

from noise cimport Noise


cdef class Perlin(Noise):

    cdef:
        double weight

    def __init__(self, weight=0.5, grid=4, size=256):
        super().__init__(grid, size)
        self.weight = weight

    cdef double wrap2(self, double x, double y, bint rot=False):
        cdef:
            double v = 0.0
            double cx, sy, _x, _y

        for _ in range(4):
            cx = cos(2 * pi * v) if rot else v
            sy = sin(2 * pi * v) if rot else v
            _x = x + self.weight * cx
            _y = y + self.weight * sy
            v = self.pnoise2(_x, _y)

        return v

    cdef double pnoise2(self, double x, double y):
        cdef:
            double nx, ny, fx, fy, w0, w1
            unsigned int i, j
            double[4] v
            double[2] arr_f, arr_n

        nx = floor(x)
        ny = floor(y)
        fx = x - nx
        fy = y - ny

        for j in range(2):
            for i in range(2):
                arr_n = [nx + i, ny + j]
                arr_f = [fx - i, fy - j]
                v[i + 2 * j] = self.gtable2(&arr_n, &arr_f)

        fx = self.fade(fx)
        fy = self.fade(fy)

        w0 = self.mix(v[0], v[1], fx)
        w1 = self.mix(v[2], v[3], fx)

        return 0.5 * self.mix(w0, w1, fy) + 0.5

    cdef double pnoise3(self, double x, double y, double z):
        cdef:
            double fx, fy, fz, nx, ny, nz, w0, w1
            unsigned int i, j, k
            double[8] v
            double[3] arr_f, arr_n

        nx = floor(x)
        ny = floor(y)
        nz = floor(z)
        fx = x - nx
        fy = y - ny
        fz = z - nz

        for k in range(2):
            for j in range(2):
                for i in range(2):
                    arr_n = [nx + i, ny + j, nz + k]
                    arr_f = [fx - i, fy - j, fz - k]
                    v[i + 2 * j + 4 * k] = self.gtable3(&arr_n, &arr_f) * 0.70710678

        fx = self.fade(fx)
        fy = self.fade(fy)
        fz = self.fade(fz)

        w0 = self.mix(self.mix(v[0], v[1], fx), self.mix(v[2], v[3], fx), fy)
        w1 = self.mix(self.mix(v[4], v[5], fx), self.mix(v[6], v[7], fx), fy)

        return 0.5 * self.mix(w0, w1, fz) + 0.5

    cpdef noise2(self):
        t = self.mock_time()

        arr = np.array(
            [self.pnoise2(x + t, y + t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    cpdef noise3(self):
        t = self.mock_time()

        arr = np.array(
            [self.pnoise3(x + t, y + t, t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr
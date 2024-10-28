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
    
    cdef double gtable2(self, double[2] *lattice, double[2] *p):
        cdef:
            unsigned int idx, i
            double u, v, _u, _v
            unsigned int[2] n

        for i in range(2):
            n[i] = <unsigned int>lattice[0][i]

        self.uhash22(&n)
        idx = n[0] >> 29

        u = (p[0][0] if idx < 4 else p[0][1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[0][1] if idx < 4 else p[0][0]) * 0.38268343   # 0.38268343 = sin(pi/8)
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    cdef double gtable3(self, double[3] *lattice, double[3] *p):
        cdef:
            unsigned int idx, i
            double u, v, _u, _v
            unsigned int[3] n

        for i in range(3):
            n[i] = <unsigned int>lattice[0][i]

        self.uhash33(&n)
        idx = n[0] >> 28
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
            arr_n[1] = ny + j
            arr_f[1] = fy - j

            for i in range(2):
                arr_n[0] = nx + i
                arr_f[0] = fx - i
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
            arr_n[2] = nz + k
            arr_f[2] = fz - k

            for j in range(2):
                arr_n[1] = ny + j
                arr_f[1] = fy - j

                for i in range(2):
                    arr_n[0] = nx + i
                    arr_f[0] = fx - i
                    v[i + 2 * j + 4 * k] = self.gtable3(&arr_n, &arr_f) * 0.70710678

        fx = self.fade(fx)
        fy = self.fade(fy)
        fz = self.fade(fz)
        w0 = self.mix(self.mix(v[0], v[1], fx), self.mix(v[2], v[3], fx), fy)
        w1 = self.mix(self.mix(v[4], v[5], fx), self.mix(v[6], v[7], fx), fy)

        return 0.5 * self.mix(w0, w1, fz) + 0.5

    cpdef noise2(self, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.pnoise2(x + t, y + t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    cpdef noise3(self, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.pnoise3(x + t, y + t, t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr
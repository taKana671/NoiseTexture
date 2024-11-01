# cython: language_level=3

import cython
import numpy as np
from libc.math cimport floor, pi

from cynoise.noise cimport Noise


cdef class Periodic(Noise):
    """Add periodicity to random numbers to create a periodic noise
       which ends are connected.
    """

    cdef:
        double period
    
    def __init__(self, period=4.0, grid=4, size=256):
        super().__init__(grid, size)
        self.period = period

    cdef double gtable2(self, double[2] *lattice, double[2] *p):
        cdef:
            unsigned int idx = 0
            unsigned int i
            double u, v, _u, _v

        for i in range(2):
            idx += <unsigned int>lattice[0][i]
            self.uhash11(&idx)

        idx = idx >> 29

        u = (p[0][0] if idx < 4 else p[0][1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[0][1] if idx < 4 else p[0][0]) * 0.38268343   # 0.38268343 = sin(pi/8)
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    cdef double gtable3(self, double[3] *lattice, double[3] *p):
        cdef:
            unsigned idx = 0
            unsigned int i
            double u, v, _u, _v

        for i in range(3):
            idx += <unsigned int>lattice[0][i]
            self.uhash11(&idx)

        idx = idx >> 28
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

    @cython.cdivision(True)
    cdef double periodic2(self, double x, double y, double t):
        cdef:
            double fx, fy, nx, ny, w0, w1, px, py, hx, hy
            unsigned int i, j
            double half_grid = <double>self.grid / 2
            double half_period = <double>self.period / 2
            double[4] v
            double[2] arr_f, arr_n

        px, py = self.xy2pol(x - half_grid, y - half_grid)
        hx = (half_period / pi) * px + t
        hy = half_period * py + t

        nx = floor(hx)
        ny = floor(hy)
        fx = hx - nx
        fy = hy - ny

        for j in range(2):
            arr_n[1] = (ny + j) % self.period
            arr_f[1] = fy - j

            for i in range(2):
                arr_n[0] = (nx + i) % self.period
                arr_f[0] = fx - i
                v[i + 2 * j] = self.gtable2(&arr_n, &arr_f)

        fx = self.fade(fx)
        fy = self.fade(fy)
        w0 = self.mix(v[0], v[1], fx)
        w1 = self.mix(v[2], v[3], fx)

        return 0.5 * self.mix(w0, w1, fy) + 0.5

    @cython.cdivision(True)
    cdef double periodic3(self, double x, double y, double t):
        cdef:
            double fx, fy, fz, nx, ny, nz, w0, w1, px, py, hx, hy
            unsigned int i, j, k
            double half_grid = <double>self.grid / 2
            double half_period = <double>self.period / 2
            double[8] v
            double[3] arr_f, arr_n

        px, py = self.xy2pol(x - half_grid, y - half_grid)
        hx = (half_period / pi) * px + t
        hy = half_period * py + t

        nx = floor(hx)
        ny = floor(hy)
        nz = floor(t)
        fx = hx - nx
        fy = hy - ny
        fz = t - nz

        for k in range(2):
            arr_n[2] = (nz + k) % self.period
            arr_f[2] = fz - k

            for j in range(2):
                arr_n[1] = (ny + j) % self.period
                arr_f[1] = fy - j

                for i in range(2):
                    arr_n[0] = (nx + i) % self.period
                    arr_f[0] = fx - i
                    v[i + 2 * j + 4 * k] = self.gtable3(&arr_n, &arr_f) * 0.70710678

        fx = self.fade(fx)
        fy = self.fade(fy)
        fz = self.fade(fz)
        w0 = self.mix(self.mix(v[0], v[1], fx), self.mix(v[2], v[3], fx), fy)
        w1 = self.mix(self.mix(v[4], v[5], fx), self.mix(v[6], v[7], fx), fy)

        return 0.5 * self.mix(w0, w1, fz) + 0.5

    cpdef noise2(self, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self.periodic2(x, y, t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    cpdef noise3(self, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self.periodic3(x, y, t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr
# cython: language_level=3

import cython
import numpy as np
from libc.math cimport floor, pi, modf

from cynoise.noise cimport Noise


cdef class PeriodicNoise(Noise):
    """Add periodicity to random numbers to create a periodic noise
       which ends are connected.
    """

    cdef:
        double period
        double half_period

    def __init__(self, period=8.0):
        super().__init__()
        self.period = period
        self.half_period = period * 0.5

    cdef double _gtable2(self, double[2] *lattice, double[2] *p):
        cdef:
            unsigned int i, n
            double u, v, _u, _v
            unsigned idx = 0

        for i in range(2):
            n = <unsigned int>lattice[0][i]
            idx = self.uhash11(idx + n)

        idx = idx >> 29

        u = (p[0][0] if idx < 4 else p[0][1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[0][1] if idx < 4 else p[0][0]) * 0.38268343   # 0.38268343 = sin(pi/8)
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    cdef double _gtable3(self, double[3] *lattice, double[3] *p):
        cdef:
            unsigned int i, n
            double u, v, _u, _v
            unsigned idx = 0

        for i in range(3):
            n = <unsigned int>lattice[0][i]
            idx = self.uhash11(idx + n)

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
    cdef double _periodic2(self, double x, double y, double t=0.0):
        cdef:
            double fx, fy, nx, ny, w0, w1, px, py, hx, hy
            unsigned int i, j
            double[4] v
            double[2] arr_f, arr_n

        px, py = self.xy2pol(2.0 * x - 1, 2.0 * y - 1)
        hx = (self.half_period / pi) * px + t
        hy = self.half_period * py + t

        nx = floor(hx)
        ny = floor(hy)
        fx = hx - nx
        fy = hy - ny

        for j in range(2):
            arr_n[1] = self.mod(ny + j, self.period)
            arr_f[1] = fy - j

            for i in range(2):
                # % is compiled to C fuction fmod(), which result differs from that of %.
                # If using %, -2 % 4 = 2, but fmod(), -2 % 4 = -2. 
                # So defined mod method in Nosie class to make -2 % 4 = 2.
                arr_n[0] = self.mod(nx + i, self.period)
                arr_f[0] = fx - i
                v[i + 2 * j] = self._gtable2(&arr_n, &arr_f)

        fx = self.quintic_hermite_interpolation(fx)
        fy = self.quintic_hermite_interpolation(fy)
        w0 = self.mix(v[0], v[1], fx)
        w1 = self.mix(v[2], v[3], fx)

        return 0.5 * self.mix(w0, w1, fy) + 0.5

    @cython.cdivision(True)
    cdef double _periodic3(self, double x, double y, double z):
        cdef:
            double fx, fy, fz, nx, ny, nz, w0, w1, px, py, hx, hy
            unsigned int i, j, k
            double[8] v
            double[3] arr_f, arr_n

        px, py = self.xy2pol(2.0 * x - 1, 2.0 * y - 1)
        hx = (self.half_period / pi) * px
        hy = self.half_period * py

        nx = floor(hx)
        ny = floor(hy)
        nz = floor(z)
        fx = hx - nx
        fy = hy - ny
        fz = z - nz

        for k in range(2):
            arr_n[2] = self.mod(nz + k, self.period)
            arr_f[2] = fz - k

            for j in range(2):
                arr_n[1] = self.mod(ny + j, self.period)
                arr_f[1] = fy - j

                for i in range(2):
                    arr_n[0] = self.mod(nx + i, self.period)
                    arr_f[0] = fx - i
                    v[i + 2 * j + 4 * k] = self._gtable3(&arr_n, &arr_f) * 0.70710678

        fx = self.quintic_hermite_interpolation(fx)
        fy = self.quintic_hermite_interpolation(fy)
        fz = self.quintic_hermite_interpolation(fz)

        w0 = self.mix(self.mix(v[0], v[1], fx), self.mix(v[2], v[3], fx), fy)
        w1 = self.mix(self.mix(v[4], v[5], fx), self.mix(v[6], v[7], fx), fy)

        return 0.5 * self.mix(w0, w1, fz) + 0.5

    cpdef double periodic2(self, double x, double y, double t=0):
        return self._periodic2(x, y, t)
    
    cpdef double periodic3(self, double x, double y, double z):
        return self._periodic3(x, y, z)
    
    cpdef noise2(self, size=256, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self._periodic2(x / size, y / size, t)
                for y in range(size)
                for x in range(size)]
        )

        arr = arr.reshape(size, size)
        return arr

    cpdef noise3(self, size=256, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self._periodic3(x / size, y / size, t)
                for y in range(size)
                for x in range(size)]
        )

        arr = arr.reshape(size, size)
        return arr
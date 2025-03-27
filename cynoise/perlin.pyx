# cython: language_level=3

import numpy as np
# cimport numpy as cnp
from libc.math cimport floor, cos, sin, pi

from cynoise.fBm cimport Fractal2D
from cynoise.noise cimport Noise
from cynoise.warping cimport DomainWarping2D


cdef class PerlinNoise(Noise):
    
    cdef double _gtable2(self, double[2] *lattice, double[2] *p):
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

    cdef double _gtable3(self, double[3] *lattice, double[3] *p):
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

    cdef double _pnoise2(self, double x, double y):
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
                v[i + 2 * j] = self._gtable2(&arr_n, &arr_f)

        fx = self.quintic_hermite_interpolation(fx)
        fy = self.quintic_hermite_interpolation(fy)
        w0 = self.mix(v[0], v[1], fx)
        w1 = self.mix(v[2], v[3], fx)

        return 0.5 * self.mix(w0, w1, fy) + 0.5

    cdef double _pnoise3(self, double x, double y, double z):
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
                    v[i + 2 * j + 4 * k] = self._gtable3(&arr_n, &arr_f) * 0.70710678

        fx = self.quintic_hermite_interpolation(fx)
        fy = self.quintic_hermite_interpolation(fy)
        fz = self.quintic_hermite_interpolation(fz)
        w0 = self.mix(self.mix(v[0], v[1], fx), self.mix(v[2], v[3], fx), fy)
        w1 = self.mix(self.mix(v[4], v[5], fx), self.mix(v[6], v[7], fx), fy)

        return 0.5 * self.mix(w0, w1, fz) + 0.5

    cpdef double pnoise3(self, double x, double y, double z):
        return self._pnoise3(x, y, z)

    cpdef double pnoise2(self, double x, double y):
        return self._pnoise2(x, y)

    cpdef noise2(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self._pnoise2(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    cpdef noise3(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self._pnoise3(x + t, y + t, t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    cpdef fractal2(self, size=256, grid=4, t=None, gain=0.5, lacunarity=2.01, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal2D(self._pnoise2, gain, lacunarity, octaves)

        arr = np.array(
            [noise._fractal2(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    cpdef warp2_rot(self, size=256, grid=4, t=None, weight=1, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal2D(self._pnoise2)
        warp = DomainWarping2D(noise._fractal2, weight=weight, octaves=octaves)

        arr = np.array(
            [warp._warp2_rot(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    cpdef warp2(self, size=256, grid=4, t=None, octaves=4):
        t = self.mock_time() if t is None else t
        weight = abs(t % 10 - 5.0)
        noise = Fractal2D(self._pnoise2)
        warp = DomainWarping2D(noise._fractal2, weight=weight, octaves=octaves)

        arr = np.array(
            [warp._warp2(x, y)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr
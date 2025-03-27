# cython: language_level=3

import cython
import numpy as np
from libc.math cimport floor

from cynoise.fBm cimport Fractal2D
from cynoise.noise cimport Noise
from cynoise.warping cimport DomainWarping2D


cdef class ValueNoise(Noise):

    cdef double _vnoise2(self, double x, double y):
        cdef:
            double nx, ny, fx, fy, w0, w1
            unsigned int i, j
            double[4] v
            double[2] arr_n

        nx = floor(x)
        ny = floor(y)

        for j in range(2):
            arr_n[1] = ny + j

            for i in range(2):
                arr_n[0] = nx + i
                v[i + 2 * j] = self.hash21(&arr_n)
        
        fx = x - nx
        fy = y - ny

        fx = self.quintic_hermite_interpolation(fx)
        fy = self.quintic_hermite_interpolation(fy)
        w0 = self.mix(v[0], v[1], fx)
        w1 = self.mix(v[2], v[3], fx)

        return self.mix(w0, w1, fy)

    cdef double _vnoise3(self, double x, double y, double z):
        cdef:
            double nx, ny, nz, fx, fy, fz, w0, w1
            unsigned int i, j, k
            double[8] v
            double[3] arr_n
            
        nx = floor(x)
        ny = floor(y)
        nz = floor(z)

        for k in range(2):
            arr_n[2] = nz + k

            for j in range(2):
                arr_n[1] = ny + j

                for i in range(2):
                    arr_n[0] = nx + i
                    v[i + 2 * j + 4 * k] = self.hash31(&arr_n)

        fx = x - nx
        fy = y - ny
        fz = z - nz

        fx = self.quintic_hermite_interpolation(fx)
        fy = self.quintic_hermite_interpolation(fy)
        fz = self.quintic_hermite_interpolation(fz)
        w0 = self.mix(self.mix(v[0], v[1], fx), self.mix(v[2], v[3], fx), fy)
        w1 = self.mix(self.mix(v[4], v[5], fx), self.mix(v[6], v[7], fx), fy)

        return self.mix(w0, w1, fz)

    @cython.cdivision(True)
    cdef double _vgrad2(self, double x, double y):
        cdef:
            double eps = 0.001
            double xx, yy
            double[2] p1 = [1.0, 1.0]
            double[2] p2

        xx = self._vnoise2(x + eps, y) - self._vnoise2(x - eps, y)
        yy = self._vnoise2(x, y + eps) - self._vnoise2(x, y - eps)
        p2[0] = 0.5 * xx / eps
        p2[1] = 0.5 * yy / eps

        return self.inner_product22(&p1, &p2)

    cpdef double vnoise2(self, double x, double y):
        return self._vnoise2(x, y)

    cpdef double vnoise3(self, double x, double y, double z):
        return self._vnoise3(x, y, z)
    
    cpdef double vgrad2(self, double x, double y):
        return self._vgrad2(x, y)

    def noise2(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self._vnoise2(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise3(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self._vnoise3(x + t, y + t, t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def grad2(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self._vgrad2(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def fractal2(self, size=256, grid=4, t=None, gain=0.5, lacunarity=2.01, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal2D(self._vnoise2, gain, lacunarity, octaves)

        arr = np.array(
            [noise._fractal2(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def warp2_rot(self, size=256, grid=4, t=None, weight=1, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal2D(self._vnoise2)
        warp = DomainWarping2D(noise._fractal2, weight, octaves)

        arr = np.array(
            [warp._warp2_rot(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def warp2(self, size=256, grid=4, octaves=4, t=None):
        t = self.mock_time() if t is None else t
        weight = abs(t % 10 - 5.0)
        noise = Fractal2D(self._vnoise2)
        warp = DomainWarping2D(noise._fractal2, weight=weight)

        arr = np.array(
            [warp._warp2(x, y)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr
# cython: profile=True
# cython: language_level=3

import math
import numpy as np
cimport numpy as cnp

from noise cimport Noise


cdef class Perlin(Noise):

    cdef:
        int grid
        int size
        double weight

    def __init__(self, weight=0.5, grid=4, size=256):
        super().__init__()
        self.size = size
        self.grid = grid
        self.weight = weight

    def pnoise2(self, p):
        n = np.floor(p)
        f, _ = np.modf(p)

        # v = [self.gtable2(n + (arr := np.array([i, j])), f - arr)
        #      for j in range(2) for i in range(2)]
        v = [self.gtable2(n + np.array([i, j]), f - np.array([i, j]))
             for j in range(2) for i in range(2)]

        f = 6 * f**5 - 15 * f**4 + 10 * f**3
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    cdef double pnoise3(self, double x, double y, double z):

        cdef:
            double fx, fy, fz, ret, w0, w1
            unsigned int i, j, k, ix, iy, iz
            double [8] v
            unsigned int[3] arr_i
            double[3] arr_f 
            

        ix = math.floor(x)
        iy = math.floor(y)
        iz = math.floor(z)

        fx = x - ix
        fy = y - iy
        fz = z - iz

        for k in range(2):
            for j in range(2):
                for i in range(2):
                    
                    arr_i = [ix + i, iy + j, iz + k]
                    arr_f = [fx - i, fy - j, fz - k]

                    ret = self.gtable3(&arr_i, &arr_f) * 0.70710678
                    v[i + 2 * j + 4 * k] = ret


        # f = 6 * f**5 - 15 * f**4 + 10 * f**3
        fx = self.fade(fx)
        fy = self.fade(fy)
        fz = self.fade(fz)

        w0 = self.mix(self.mix(v[0], v[1], fx), self.mix(v[2], v[3], fx), fy)
        w1 = self.mix(self.mix(v[4], v[5], fx), self.mix(v[6], v[7], fx), fy)
        return 0.5 * self.mix(w0, w1, fz) + 0.5

    def noise2(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.pnoise2(np.array([x + t, y + t]))
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    cpdef noise3(self, t=None):
        t = self.mock_time() if t is None else t
        # self.hash = {}

        arr = np.array(
            [self.pnoise3(x + t, y + t, t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def wrap2(self, x, y, rot=False):
        v = 0.0

        for i in range(4):
            cx = np.cos(2 * np.pi * v) if rot else v
            sy = np.sin(2 * np.pi * v) if rot else v
            _x = x + self.weight * cx
            _y = y + self.weight * sy
            v = self.pnoise2(np.array([_x, _y]))

        return v
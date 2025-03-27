# cython: language_level=3

import cython
import numpy as np
from libc.math cimport floor, cos, sin, pi, fmax, fmin, modf, fabs

from cynoise.fBm cimport Fractal2D
from cynoise.noise cimport Noise
from cynoise.warping cimport DomainWarping2D


cdef class SimplexNoise(Noise):

    #cdef void add22(self, double[2] *arr1, double[2] *arr2, double[2] *result):
    #    result[0][0] = arr1[0][0] + arr2[0][0]
    #    result[0][1] = arr1[0][1] + arr2[0][1]
    
    #cdef void floor2(self, double[2] *arr, double[2] *result):
    #    result[0][0] = floor(arr[0][0])
    #    result[0][1] = floor(arr[0][1])

    @cython.cdivision(True)
    cdef double mod289(self, double v):
        return v - floor(v * (1.0 / 289.0)) * 289.0
    
    cdef double permute(self, double v):
        return self.mod289(((v * 34.0) + 1.0) * v)

    cdef double inverssqrt(self, double v):
        return 1.79284291400159 - 0.85373472095314 * v
        # return 1 / v ** 2

    @cython.cdivision(True)
    cdef double _snoise2(self, double x, double y, int scale):
        cdef:
            double[2] p = [x * scale, y * scale]
            double[2] x0, x1, x2, i, i1, i2
            double[3] perm, ip, a0, h, mm, g
            double[4] grid
            double inner_product, f, m, iptr, xx, ox
            unsigned int idx
        
        grid[0] = (3.0 - 3.0 ** 0.5) / 6.0
        grid[1] = 0.5 * (3.0 ** 0.5 - 1.0)
        grid[2] = -1.0 + 2.0 * grid[0]
        grid[3] = 1.0 / 41.0

        # the first corner
        inner_product = p[0] * grid[1] + p[1] * grid[1] 
        for idx in range(2):
            i[idx] = floor(p[idx] + inner_product)
        
        inner_product = i[0] * grid[0] + i[1] * grid[0] 
        for idx in range(2):
            x0[idx] = p[idx] - i[idx] + inner_product

        # the other two corners
        if x0[0] > x0[1]:
            i1 = [1.0, 0.0]
        else:
            i1 = [0.0, 1.0]

        for idx in range(2):
            x1[idx] = x0[idx] + grid[0] - i1[idx]
            x2[idx] = x0[idx] + grid[2]

        perm = [0, 0, 0]
        for idx in range(1, -1, -1):
            m = self.mod289(i[idx])
            perm[0] = self.permute(perm[0] + m)
            perm[1] = self.permute(perm[1] + m + i1[idx])
            perm[2] = self.permute(perm[2] + m + 1.0)
        
        ip = [
            self.inner_product22(&x0, &x0),
            self.inner_product22(&x1, &x1),
            self.inner_product22(&x2, &x2),
        ]

        for idx in range(3):
            m = fmax(0.5 - ip[idx], 0.0)
            m = m ** 4

            f = modf(perm[idx] * grid[3], &iptr)
            xx = 2.0 * f - 1.0
            h[idx] = fabs(xx) - 0.5
            ox = floor(xx + 0.5)
            a0[idx] = xx - ox
            mm[idx] = m * self.inverssqrt(a0[idx] ** 2 + h[idx] ** 2)
        
        g = [
            a0[0] * x0[0] + h[0] * x0[1],
            a0[1] * x1[0] + h[1] * x1[1],
            a0[2] * x2[0] + h[2] * x2[1]
        ]

        return 130.0 * self.inner_product33(&mm, &g) * 0.5 + 0.5

    cdef void product31(self, double[3] *arr, double v):
        cdef:
            unsigned int i
            double elem
        
        for i in range(3):
            elem = arr[0][i]
            arr[0][i] = elem * v
            # arr[0][i] = arr[0][i] * multiplier

    @cython.cdivision(True)
    cdef double _snoise3(self, double x, double y, double z, int scale):
        cdef:
            double[3] p = [x * scale, y * scale, z * scale]
            double[2] c = [1.0 / 6.0, 1.0 / 3.0]
            double[4] d = [0.0, 0.5, 1.0, 2.0]
            double[4] perm, xx, yy, h, b0, b1, s0, s1, sh, a0, a1, mx, cp
            double[3] i, i1, i2, x0, x1, x2, x3, ll, g, tmp, ns, p0, p1, p2, p3
            double inner_prod1, inner_prod2, m, div, _xx, _yy, j
            unsigned int k
        
        # the first corner
        inner_prod1 = self.inner_product31(&p, c[1])
        inner_prod2 = 0.0

        for k in range(3):
            i[k] = floor(p[k] + inner_prod1)
            inner_prod2 += i[k] * c[0]
        
        for k in range(3):
            x0[k] = p[k] - i[k] + inner_prod2
        
        # other corners
        tmp = [x0[1], x0[2], x0[0]]

        for k in range(3):
            g[k] = <double>self.step(tmp[k], x0[k])
            ll[k] = 1.0 - g[k]
        
        tmp = [ll[2], ll[0], ll[1]]

        for k in range(3):
            i1[k] = fmin(g[k], tmp[k])
            i2[k] = fmax(g[k], tmp[k])
            x1[k] = x0[k] - i1[k] + c[0]
            x2[k] = x0[k] - i2[k] + c[1]
            x3[k] = x0[k] - d[1]

        perm = [0, 0, 0, 0]

        for k in range(2, -1, -1):
            m = self.mod289(i[k])
            perm[0] = self.permute(perm[0] + m)
            perm[1] = self.permute(perm[1] + m + i1[k])
            perm[2] = self.permute(perm[2] + m + i2[k])
            perm[3] = self.permute(perm[3] + m + 1.0)
        
        div = 1.0 / 7.0
        ns = [div * d[3] - d[0], div * d[1] - d[2], div * d[2] - d[0]]

        for k in range(4):
            j = perm[k] - 49.0 * floor(perm[k] * ns[2] * ns[2])
            _xx = floor(j * ns[2])
            _yy = floor(j - 7.0 * _xx)
            xx[k] = _xx * ns[0] + ns[1]
            yy[k] = _yy * ns[0] + ns[1]
            h[k] = 1.0 - fabs(xx[k]) - fabs(yy[k])
        
        b0 = [xx[0], xx[1], yy[0], yy[1]]
        b1 = [xx[2], xx[3], yy[2], yy[3]]

        for k in range(4):
            s0[k] = floor(b0[k]) * 2.0 + 1.0
            s1[k] = floor(b1[k]) * 2.0 + 1.0
            sh[k] = <double>self.step(h[k], 0.0) * -1.0

        a0 = [b0[0] + s0[0] * sh[0], b0[2] + s0[2] * sh[0],
              b0[1] + s0[1] * sh[1], b0[3] + s0[3] * sh[1]]

        a1 = [b1[0] + s1[0] * sh[2], b1[2] + s1[2] * sh[2],
              b1[1] + s1[1] * sh[3], b1[3] + s1[3] * sh[3]]

        p0 = [a0[0], a0[1], h[0]]
        p1 = [a0[2], a0[3], h[1]]
        p2 = [a1[0], a1[1], h[2]]
        p3 = [a1[2], a1[3], h[3]]
            
        self.product31(&p0, self.inverssqrt(self.inner_product33(&p0, &p0)))
        self.product31(&p1, self.inverssqrt(self.inner_product33(&p1, &p1)))
        self.product31(&p2, self.inverssqrt(self.inner_product33(&p2, &p2)))
        self.product31(&p3, self.inverssqrt(self.inner_product33(&p3, &p3)))

        mx = [
            fmax(0.6 - self.inner_product33(&x0, &x0), 0.0) ** 4,
            fmax(0.6 - self.inner_product33(&x1, &x1), 0.0) ** 4,
            fmax(0.6 - self.inner_product33(&x2, &x2), 0.0) ** 4,
            fmax(0.6 - self.inner_product33(&x3, &x3), 0.0) ** 4
        ]

        cp = [
            self.inner_product33(&p0, &x0),
            self.inner_product33(&p1, &x1),
            self.inner_product33(&p2, &x2),
            self.inner_product33(&p3, &x3)
        ]

        return 42.0 * self.inner_product44(&mx, &cp) * 0.5 + 0.5

    cpdef noise2(self, width=256, height=256, scale=20, t=None):
        t = self.mock_time() if t is None else t
        m = min(width, height)

        arr = np.array(
            [self._snoise2(x / m + t,  y / m + t, scale)
                for y in range(height)
                for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr
    
    def noise3(self, width=256, height=256, scale=20, t=None):
        t = self.mock_time() if t is None else t
        m = min(width, height)

        arr = np.array(
            [self._snoise3(x / m + t, y / m + t, t, scale)
                for y in range(height)
                for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr




        



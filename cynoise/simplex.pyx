# cython: language_level=3

import cython
import numpy as np
from libc.math cimport floor, fmax, fmin, modf, fabs

from .fBm cimport Fractal2D, Fractal3D
from .noise cimport Noise


cdef class SimplexNoise(Noise):

    @cython.cdivision(True)
    cdef double mod289(self, double v):
        return v - floor(v * (1.0 / 289.0)) * 289.0
    
    cdef double permute(self, double v):
        return self.mod289(((v * 34.0) + 1.0) * v)

    cdef double inverssqrt(self, double v):
        return 1.79284291400159 - 0.85373472095314 * v
        # return 1 / v ** 2

    @cython.cdivision(True)
    cdef double _snoise2(self, double x, double y):
        cdef:
            double[2] p = [x, y]
            double[3][2] cn
            double[2][2] dr
            double[3] perm, a0, h, mm, g
            double[4] grid
            double ip, f, m, iptr, xx, ox
            unsigned int i
        
        grid[0] = (3.0 - 3.0 ** 0.5) / 6.0
        grid[1] = 0.5 * (3.0 ** 0.5 - 1.0)
        grid[2] = -1.0 + 2.0 * grid[0]
        grid[3] = 1.0 / 41.0

        # the first corner 
        ip = self.inner_product21(&p, &grid[1])

        dr[0][0] = floor(p[0] + ip)
        dr[0][1] = floor(p[1] + ip)
        
        ip = self.inner_product21(&dr[0], &grid[0])

        cn[0][0] = p[0] - dr[0][0] + ip
        cn[0][1] = p[1] - dr[0][1] + ip

        # the other two corners
        if cn[0][0] > cn[0][1]:
            dr[1][:] = [1.0, 0.0]
        else:
            dr[1][:] = [0.0, 1.0]

        for i in range(2):
            cn[1][i] = cn[0][i] + grid[0] - dr[1][i]
            cn[2][i] = cn[0][i] + grid[2]

        perm = [0, 0, 0]

        for i in range(1, -1, -1):
            m = self.mod289(dr[0][i])
            perm[0] = self.permute(perm[0] + m)
            perm[1] = self.permute(perm[1] + m + dr[1][i])
            perm[2] = self.permute(perm[2] + m + 1.0)
        
        for i in range(3):
            ip = self.inner_product22(&cn[i], &cn[i])
            m = fmax(0.5 - ip, 0.0) ** 4
            f = modf(perm[i] * grid[3], &iptr)
            xx = 2.0 * f - 1.0
            h[i] = fabs(xx) - 0.5
            ox = floor(xx + 0.5)
            a0[i] = xx - ox
            mm[i] = m * self.inverssqrt(a0[i] ** 2 + h[i] ** 2)
        
        g = [
            a0[0] * cn[0][0] + h[0] * cn[0][1],
            a0[1] * cn[1][0] + h[1] * cn[1][1],
            a0[2] * cn[2][0] + h[2] * cn[2][1]
        ]

        return 130.0 * self.inner_product33(&mm, &g) * 0.5 + 0.5

    cdef void product31(self, double[3] *arr, double v):
        cdef:
            unsigned int i
            double elem
        
        for i in range(3):
            elem = arr[0][i]
            arr[0][i] = elem * v

    @cython.cdivision(True)
    cdef double _snoise3(self, double x, double y, double z):
        cdef:
            double[3] p = [x, y, z]
            double[2] c = [1.0 / 6.0, 1.0 / 3.0]
            double[4] d = [0.0, 0.5, 1.0, 2.0]
            double[4] perm, h, sh, mx, cp
            double[3] ll, g, tmp, ns
            double[2][4] a, b, s, t
            double[4][3] pt, cn
            double[3][3] dr

            double ip, m, div, _x, _y, j
            unsigned int i
        
        # the first corner
        ip = self.inner_product31(&p, &c[1])

        for i in range(3):
            dr[0][i] = floor(p[i] + ip)

        ip = self.inner_product31(&dr[0], &c[0])

        for i in range(3):
            cn[0][i] = p[i] - dr[0][i] + ip
        
        # other corners
        tmp = [cn[0][1], cn[0][2], cn[0][0]]

        for i in range(3):
            g[i] = <double>self.step(tmp[i], cn[0][i])
            ll[i] = 1.0 - g[i]
        
        tmp = [ll[2], ll[0], ll[1]]

        for i in range(3):
            dr[1][i] = fmin(g[i], tmp[i])
            dr[2][i] = fmax(g[i], tmp[i])
            cn[1][i] = cn[0][i] - dr[1][i] + c[0]
            cn[2][i] = cn[0][i] - dr[2][i] + c[1]
            cn[3][i] = cn[0][i] - d[1]

        perm = [0, 0, 0, 0]

        for i in range(2, -1, -1):
            m = self.mod289(dr[0][i])
            perm[0] = self.permute(perm[0] + m)
            perm[1] = self.permute(perm[1] + m + dr[1][i])
            perm[2] = self.permute(perm[2] + m + dr[2][i])
            perm[3] = self.permute(perm[3] + m + 1.0)
        
        div = 1.0 / 7.0
        ns = [div * d[3] - d[0], div * d[1] - d[2], div * d[2] - d[0]]

        for i in range(4):
            j = perm[i] - 49.0 * floor(perm[i] * ns[2] * ns[2])
            _x = floor(j * ns[2])
            _y = floor(j - 7.0 * _x)
            t[0][i] = _x * ns[0] + ns[1]
            t[1][i] = _y * ns[0] + ns[1]
            h[i] = 1.0 - fabs(t[0][i]) - fabs(t[1][i])
        
        b[0][:] = [t[0][0], t[0][1], t[1][0], t[1][1]]
        b[1][:] = [t[0][2], t[0][3], t[1][2], t[1][3]]

        for i in range(4):
            s[0][i] = floor(b[0][i]) * 2.0 + 1.0
            s[1][i] = floor(b[1][i]) * 2.0 + 1.0
            sh[i] = <double>self.step(h[i], 0.0) * -1.0

        a[0][:] = [b[0][0] + s[0][0] * sh[0], b[0][2] + s[0][2] * sh[0],
                   b[0][1] + s[0][1] * sh[1], b[0][3] + s[0][3] * sh[1]]

        a[1][:] = [b[1][0] + s[1][0] * sh[2], b[1][2] + s[1][2] * sh[2], 
                   b[1][1] + s[1][1] * sh[3], b[1][3] + s[1][3] * sh[3]]

        pt[0][:] = [a[0][0], a[0][1], h[0]]
        pt[1][:] = [a[0][2], a[0][3], h[1]]
        pt[2][:] = [a[1][0], a[1][1], h[2]]
        pt[3][:] = [a[1][2], a[1][3], h[3]]
            
        self.product31(&pt[0], self.inverssqrt(self.inner_product33(&pt[0], &pt[0])))
        self.product31(&pt[1], self.inverssqrt(self.inner_product33(&pt[1], &pt[1])))
        self.product31(&pt[2], self.inverssqrt(self.inner_product33(&pt[2], &pt[2])))
        self.product31(&pt[3], self.inverssqrt(self.inner_product33(&pt[3], &pt[3])))

        mx = [
            fmax(0.6 - self.inner_product33(&cn[0], &cn[0]), 0.0) ** 4,
            fmax(0.6 - self.inner_product33(&cn[1], &cn[1]), 0.0) ** 4,
            fmax(0.6 - self.inner_product33(&cn[2], &cn[2]), 0.0) ** 4,
            fmax(0.6 - self.inner_product33(&cn[3], &cn[3]), 0.0) ** 4
        ]

        cp = [
            self.inner_product33(&pt[0], &cn[0]),
            self.inner_product33(&pt[1], &cn[1]),
            self.inner_product33(&pt[2], &cn[2]),
            self.inner_product33(&pt[3], &cn[3])
        ]

        return 42.0 * self.inner_product44(&mx, &cp) * 0.5 + 0.5

    cpdef double snoise2(self, double x, double y):
        return self._snoise2(x, y)
    
    cpdef double snoise3(self, double x, double y, double z):
        return self._snoise3(x, y, z)
     
    cpdef noise2(self, width=256, height=256, scale=20.0, t=None):
        t = self.mock_time() if t is None else t
        m = min(width, height)

        arr = np.array(
            [self._snoise2((x / m + t) * scale,  (y / m + t) * scale)
                for y in range(height)
                for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr
    
    cpdef noise3(self, width=256, height=256, scale=20.0, t=None):
        t = self.mock_time() if t is None else t
        m = min(width, height)

        arr = np.array(
            [self._snoise3((x / m + t) * scale, (y / m + t) * scale, t * scale)
                for y in range(height)
                for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr
    
    cpdef fractal2(self, width=256, height=256, t=None,
                   gain=0.5, lacunarity=2.01, octaves=4):
        t = self.mock_time() if t is None else t
        m = min(width, height)

        noise = Fractal2D(
            self._snoise2, gain=gain, lacunarity=lacunarity, octaves=octaves)

        arr = np.array(
            [noise._fractal2(x / m + t, y / m + t)
             for y in range(height) for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr

    cpdef fractal3(self, width=256, height=256, t=None,
                   gain=0.5, lacunarity=2.01, octaves=4):
        t = self.mock_time() if t is None else t
        m = min(width, height)

        noise = Fractal3D(
            self._snoise3, gain=gain, lacunarity=lacunarity, octaves=octaves)

        arr = np.array(
            [noise._fractal3(x / m + t, y / m + t, t)
             for y in range(height) for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr




        



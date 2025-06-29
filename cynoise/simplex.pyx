# cython: language_level=3

import cython
import numpy as np
from libc.math cimport floor, fmax, fmin, modf, fabs, cos, sin, pi

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

    @cython.cdivision(True)
    cdef double _snoise4(self, double x, double y, double z, double w):
        cdef:
            double[4] grid
            double[4] p = [x, y, z, w]
            double f4 = (5.0 ** 0.5 - 1.0) / 4.0
            double j0 = 0.0
            double[4] j1 = [0.0, 0.0, 0.0, 0.0]

            double inner, v, norm
            double[2] m1, a1
            double[3] ip, m0, a0
            double[4] d0
            double[5][4] xn, pn
            double[4][4] ni
            unsigned int i, j

        grid[0] = (5.0 - 5.0 ** 0.5) / 20.0
        grid[1] = 2.0 * grid[0]
        grid[2] = 3.0 * grid[0]
        grid[3] = -1.0 + 4.0 * grid[0]

        inner = self.inner_product41(&p, &f4)
        for i in range(4):
            d0[i] = floor(p[i] + inner)
        
        inner = self.inner_product41(&d0, &grid[0])
        for i in range(4):
            xn[0][i] = p[i] - d0[i] + inner

        ni[0][0] = 0.0
        for i in range(3):
            v = self.step(xn[0][i + 1], xn[0][0])
            ni[0][0] += v
            ni[0][i + 1] = 1.0 - v
        
        for i in range(2):
            v = self.step(xn[0][i + 2], xn[0][1])
            ni[0][1] += v
            ni[0][i + 2] += 1.0 - v

        v = self.step(xn[0][3], xn[0][2])
        ni[0][2] += v
        ni[0][3] += 1 - v

        for i in range(4):
            ni[3][i] = self.clamp(ni[0][i], 0.0, 1.0)
            ni[2][i] = self.clamp(ni[0][i] - 1.0, 0.0, 1.0)
            ni[1][i] = self.clamp(ni[0][i] - 2.0, 0.0, 1.0)
        
            xn[1][i] = xn[0][i] - ni[1][i] + grid[0]
            xn[2][i] = xn[0][i] - ni[2][i] + grid[1]
            xn[3][i] = xn[0][i] - ni[3][i] + grid[2]
            xn[4][i] = xn[0][i] + grid[3]
        
        for i in range(3, -1, -1):
            m = self.mod289(d0[i])
            j0 = self.permute(m + j0)
            j1[0] = self.permute(m + ni[1][i] + j1[0])
            j1[1] = self.permute(m + ni[2][i] + j1[1])
            j1[2] = self.permute(m + ni[3][i] + j1[2])
            j1[3] = self.permute(m + 1.0 + j1[3])

        ip = [1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0]

        for i in range(5):
            v = j0 if i == 0 else j1[i - 1]
            self.grad4(&v, &ip, &pn[i])

            inner = self.inner_product44(&pn[i], &pn[i])
            norm = self.inverssqrt(inner)

            for j in range(4):
                pn[i][j] *= norm
  
        for i in range(5):
            v = fmax(0.6 - self.inner_product44(&xn[i], &xn[i]), 0.0)
            inner = self.inner_product44(&pn[i], &xn[i])

            if i < 3:
                m0[i] = v ** 4
                a0[i] = inner
            else:
                m1[i-3] = v ** 4
                a1[i-3] = inner

        return 49.0 * (self.inner_product33(&m0, &a0) + self.inner_product22(&m1, &a1)) * 0.5 + 0.5

    cdef void grad4(self, double *j, double[3] *ip, double[4] *p):
        cdef:
            double iptr, f, v, s
            unsigned int i

        v = 0.0
        for i in range(3):
            f = modf(j[0] * ip[0][i], &iptr)
            p[0][i] = floor(f * 7.0) * ip[0][2] - 1.0
            v += fabs(p[0][i])

        p[0][3] = 1.5 - v

        s = 1 if p[0][3] < 0 else 0
        for i in range(3):
            v = 1 if p[0][i] < 0 else 0
            p[0][i] += (v * 2.0 - 1.0) * s

    cpdef double snoise2(self, double x, double y):
        return self._snoise2(x, y)
    
    cpdef double snoise3(self, double x, double y, double z):
        return self._snoise3(x, y, z)

    cpdef double snoise4(self, double x, double y, double z, double w):
        return self._snoise4(x, y, z, w)
     
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

    cpdef noise4(self, width=256, height=256, scale=20, t=None):
        t = self.mock_time() if t is None else float(t)
        m = min(width, height)

        arr = np.array(
            [self._snoise4((x / m + t) * scale, (y / m + t) * scale, 0, t * scale)
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


cdef class TileableSimplexNoise(SimplexNoise):

    cpdef tile(self, x, y, scale=3, aa=123, bb=231, cc=321, dd=273):
        a = sin(x * 2 * pi) * scale + aa
        b = cos(x * 2 * pi) * scale + bb
        c = sin(y * 2 * pi) * scale + cc
        d = cos(y * 2 * pi) * scale + dd

        return self.snoise4(a, b, c, d)

    cpdef tileable_noise(self, width=256, height=256, scale=3, t=None, is_rnd=True):
        """Args:
            scale (float): The smaller scale is, the larger the noise spacing becomes,
                       and the larger it is, the smaller the noise spacing becomes.
        """
        t = self.mock_time() if t is None else t
        m = min(width, height)
        aa, bb, cc, dd = self.get_4_nums(is_rnd)

        arr = np.array(
            [self.tile(x / m + t, y / m + t, scale, aa, bb, cc, dd)
                for y in range(height) for x in range(width)]
        )

        arr = arr.reshape(height, width)
        return arr





        



# cython: language_level=3

import numpy as np
# cimport numpy as cnp
from libc.math cimport floor, cos, sin, pi

from .fBm cimport Fractal2D
from .noise cimport Noise
from .warping cimport DomainWarping2D


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

    cdef double _gtable4(self, double[4] *lattice, double[4] *p):
        cdef:
            unsigned int idx, i
            double t, u, v, _t, _u, _v
            unsigned int[4] n

        for i in range(4):
            n[i] = <unsigned int>lattice[0][i]
        
        self.uhash44(&n)
        idx = n[0] >> 27

        t = p[0][0] if idx < 24 else p[0][1]
        u = p[0][1] if idx < 16 else p[0][2]
        v = p[0][2] if idx < 8 else p[0][3]

        _t = t if idx & 3 == 0 else -t
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v
    
        return _t + _u + _v


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

    cdef double _pnoise4(self, double x, double y, double z, double w):
        cdef:
            double fx, fy, fz, fw, nx, ny, nz, nw, m1, m2, u0, u1
            unsigned int i, j, k, l
            double[16] v
            double[4] arr_f, arr_n, arr_w

        nx = floor(x)
        ny = floor(y)
        nz = floor(z)
        nw = floor(w)
        fx = x - nx
        fy = y - ny
        fz = z - nz
        fw = w - nw

        for l in range(2):
            arr_n[3] = nw + l
            arr_f[3] = fw - l

            for k in range(2):
                arr_n[2] = nz + k
                arr_f[2] = fz - k

                for j in range(2):
                    arr_n[1] = ny + j
                    arr_f[1] = fy - j

                    for i in range(2):
                        arr_n[0] = nx + i
                        arr_f[0] = fx - i
                        v[i + 2 * j + 4 * k + 8 * l] = self._gtable4(&arr_n, &arr_f) * 0.57735027

        fx = self.quintic_hermite_interpolation(fx)
        fy = self.quintic_hermite_interpolation(fy)
        fz = self.quintic_hermite_interpolation(fz)
        fw = self.quintic_hermite_interpolation(fw)

        for i in range(4):
            m1 = self.mix(v[4 * i], v[4 * i + 1], fx)
            m2 = self.mix(v[4 * i + 2], v[4 * i + 3], fx)
            arr_w[i] = self.mix(m1, m2, fy)

        u0 = self.mix(arr_w[0], arr_w[1], fz)
        u1 = self.mix(arr_w[2], arr_w[3], fz)

        return 0.5 * self.mix(u0, u1, fw) + 0.5

    cpdef double pnoise2(self, double x, double y):
        return self._pnoise2(x, y)

    cpdef double pnoise3(self, double x, double y, double z):
        return self._pnoise3(x, y, z)

    cpdef double pnoise4(self, double x, double y, double z, double w):
        return self._pnoise4(x, y, z, w)

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

    cpdef noise4(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self._pnoise4(x + t, y + t, 0, t)
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

    cpdef warp2_rot(self, size=256, grid=4, t=None, weight=1.0, octaves=4):
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
        mod = self.mod(t, 10)
        weight = abs(mod - 5.0)
        # weight = abs(t % 10 - 5.0) <- 't % 10' maybe causes warning C4244: '=': conversion from 'Py_ssize_t' to 'long',
        noise = Fractal2D(self._pnoise2)
        warp = DomainWarping2D(noise._fractal2, weight=weight, octaves=octaves)

        arr = np.array(
            [warp._warp2(x, y)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr


cdef class TileablePerlinNoise(PerlinNoise):

    cpdef tile(self, x, y, scale=1, aa=123, bb=231, cc=321, dd=273):
        a = sin(x * 2 * pi) * scale + aa
        b = cos(x * 2 * pi) * scale + bb
        c = sin(y * 2 * pi) * scale + cc
        d = cos(y * 2 * pi) * scale + dd

        return self._pnoise4(a, b, c, d)
    

    cpdef tileable_noise(self, size=256, scale=0.8, t=None, is_rnd=True):
        """Args:
            scale (float): The smaller scale is, the larger the noise spacing becomes,
                       and the larger it is, the smaller the noise spacing becomes.
        """
        t = self.mock_time() if t is None else float(t)
        aa, bb, cc, dd = self.get_4_nums(is_rnd)

        arr = np.array(
            [self.tile((x + t) / size, (y + t) / size, scale, aa, bb, cc, dd)
                for y in range(size) for x in range(size)]
        )

        arr = arr.reshape(size, size)
        return arr
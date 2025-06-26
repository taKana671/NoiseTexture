import numpy as np

from .fBm import Fractal2D
from .warping import DomainWarping2D
from .noise import Noise, cache


class PerlinNoise(Noise):

    @cache(128)
    def hash44(self, p):
        n = p.astype(np.uint32)
        self.uhash44(n)
        idx = n[0] >> 27

        return idx

    @cache(128)
    def hash33(self, p):
        n = p.astype(np.uint32)
        self.uhash33(n)
        idx = n[0] >> 28

        return idx

    @cache(128)
    def hash22(self, p):
        n = p.astype(np.uint32)
        self.uhash22(n)
        idx = n[0] >> 29

        return idx

    def gtable2(self, lattice, p):
        """Args:
            lattice, p (numpy.ndarray): 2-element array
        """
        idx = self.hash22(lattice)

        u = (p[0] if idx < 4 else p[1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[1] if idx < 4 else p[0]) * 0.38268343   # 0.38268343 = sin(pi/8)
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    def gtable3(self, lattice, p):
        """Args:
            lattice, p (numpy.ndarray): 3-element array
        """
        idx = self.hash33(lattice)
        u = p[0] if idx < 8 else p[1]

        if idx < 4:
            v = p[1]
        elif idx == 12 or idx == 14:
            v = p[0]
        else:
            v = p[2]

        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    def gtable4(self, lattice, p):
        idx = self.hash44(lattice)
        t = p[0] if idx < 24 else p[1]
        u = p[1] if idx < 16 else p[2]
        v = p[2] if idx < 8 else p[3]

        _t = t if idx & 3 == 0 else -t
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _t + _u + _v

    def pnoise2(self, x, y):
        """Args:
            x, y (float)
        """
        p = np.array([x, y])
        n = np.floor(p)
        f = p - n

        v = [self.gtable2(n + (arr := np.array([i, j])), f - arr)
             for j in range(2) for i in range(2)]

        f = self.quintic_hermite_interpolation(f)
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])

        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def pnoise3(self, x, y, z):
        """Args:
            x, y, z (float)
        """
        p = np.array([x, y, z])
        n = np.floor(p)
        f = p - n

        # 0.70710678; 1 / sqrt(2)
        v = [self.gtable3(n + (arr := np.array([i, j, k])), f - arr) * 0.70710678
             for k in range(2) for j in range(2) for i in range(2)]

        f = self.quintic_hermite_interpolation(f)
        w0 = self.mix(self.mix(v[0], v[1], f[0]), self.mix(v[2], v[3], f[0]), f[1])
        w1 = self.mix(self.mix(v[4], v[5], f[0]), self.mix(v[6], v[7], f[0]), f[1])

        return 0.5 * self.mix(w0, w1, f[2]) + 0.5

    def pnoise4(self, x, y, z, w):
        p = np.array([x, y, z, w])
        n = np.floor(p)
        f = p - n

        # 0.57735027; 1/sqrt(3)
        v = [self.gtable4(n + (arr := np.array([i, j, k, ll])), f - arr) * 0.57735027
             for ll in range(2) for k in range(2) for j in range(2) for i in range(2)]

        f = self.quintic_hermite_interpolation(f)

        w = []
        for i in range(4):
            m1 = self.mix(v[4 * i], v[4 * i + 1], f[0])
            m2 = self.mix(v[4 * i + 2], v[4 * i + 3], f[0])
            w.append(self.mix(m1, m2, f[1]))

        u0 = self.mix(w[0], w[1], f[2])
        u1 = self.mix(w[2], w[3], f[2])

        return 0.5 * self.mix(u0, u1, f[3]) + 0.5

    def noise2(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.pnoise2(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def noise3(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.pnoise3(x + t, y + t, t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def noise4(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.pnoise4(x + t, y + t, 0, t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def fractal2(self, size=256, grid=4, t=None, gain=0.5, lacunarity=2.01, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal2D(self.pnoise2, gain, lacunarity, octaves)

        arr = np.array(
            [noise.fractal(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def warp2_rot(self, size=256, grid=4, t=None, weight=1, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal2D(self.pnoise2)
        warp = DomainWarping2D(noise.fractal, weight, octaves)

        arr = np.array(
            [warp.warp_rot(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def warp2(self, size=256, grid=4, t=None, octaves=4):
        t = self.mock_time() if t is None else t
        weight = abs(t % 10 - 5.0)
        noise = Fractal2D(self.pnoise2)
        warp = DomainWarping2D(noise.fractal, weight, octaves)

        arr = np.array(
            [warp.warp(x, y)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr


class TileablePerlinNoise(PerlinNoise):

    def tile(self, x, y, scale=1, aa=123, bb=231, cc=321, dd=273):
        a = np.sin(x * 2 * np.pi) * scale + aa
        b = np.cos(x * 2 * np.pi) * scale + bb
        c = np.sin(y * 2 * np.pi) * scale + cc
        d = np.cos(y * 2 * np.pi) * scale + dd

        return self.pnoise4(a, b, c, d)

    def tileable_noise(self, size=256, scale=0.8, t=None, is_rnd=True):
        """Args:
            scale (float): The smaller scale is, the larger the noise spacing becomes,
                       and the larger it is, the smaller the noise spacing becomes.
        """
        t = self.mock_time() if t is None else t
        aa, bb, cc, dd = self.get_4_nums(is_rnd)

        arr = np.array(
            [self.tile((x + t) / size, (y + t) / size, scale, aa, bb, cc, dd)
                for y in range(size) for x in range(size)]
        )

        arr = arr.reshape(size, size)
        return arr
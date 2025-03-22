import numpy as np

from .fBm import Fractal
from .warping import DomainWarping
from .noise import Noise, cache


class PerlinNoise(Noise):

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
            lattice, p (numpy.ndarray): 2-dimensional array
        """
        idx = self.hash22(lattice)

        u = (p[0] if idx < 4 else p[1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[1] if idx < 4 else p[0]) * 0.38268343   # 0.38268343 = sin(pi/8)
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    def gtable3(self, lattice, p):
        """Args:
            lattice, p (numpy.ndarray): 3-dimensional array
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

    def pnoise2(self, p):
        """Args:
            p (numpy.ndarray): 2-dimensional array
        """
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable2(n + (arr := np.array([i, j])), f - arr)
             for j in range(2) for i in range(2)]

        f = self.quintic_hermite_interpolation(f)
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def pnoise3(self, p):
        """Args:
            p (numpy.ndarray): 3-dimensional array
        """
        n = np.floor(p)
        f, _ = np.modf(p)

        # 0.70710678 = 1 / sqrt(2)
        v = [self.gtable3(n + (arr := np.array([i, j, k])), f - arr) * 0.70710678
             for k in range(2) for j in range(2) for i in range(2)]

        f = self.quintic_hermite_interpolation(f)
        w0 = self.mix(self.mix(v[0], v[1], f[0]), self.mix(v[2], v[3], f[0]), f[1])
        w1 = self.mix(self.mix(v[4], v[5], f[0]), self.mix(v[6], v[7], f[0]), f[1])
        return 0.5 * self.mix(w0, w1, f[2]) + 0.5

    def noise2(self, grid=4, size=256, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.pnoise2(np.array([x, y]) + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def noise3(self, grid=4, size=256, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.pnoise3(np.array([x + t, y + t, t]))
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def fractal2(self, size=256, grid=4, t=None, gain=0.5, lacunarity=2.01, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal(self.pnoise2, gain, lacunarity, octaves)

        arr = np.array(
            [noise.fractal(np.array([x, y]) + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def warp2_rot(self, size=256, grid=4, t=None, weight=1, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal(self.pnoise2)
        warp = DomainWarping(noise.fractal, weight, octaves)

        arr = np.array(
            [warp.warp2_rot(np.array([x, y]) + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def warp2(self, size=256, grid=4, t=None, octaves=4):
        t = self.mock_time() if t is None else t
        weight = abs(t % 10 - 5.0)
        noise = Fractal(self.pnoise2)
        warp = DomainWarping(noise.fractal, weight=weight)

        arr = np.array(
            [warp.warp(np.array([x, y]))
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr
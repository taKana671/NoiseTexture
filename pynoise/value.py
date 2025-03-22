import numpy as np

from .fBm import Fractal
from .warping import DomainWarping
from .noise import Noise


class ValueNoise(Noise):

    def vnoise2(self, p):
        """Args:
            p (numpy.ndarray): 2-dimensional array
        """
        n = np.floor(p)
        v = [self.hash21(n + np.array([i, j])) for j in range(2) for i in range(2)]

        f, _ = np.modf(p)
        # f = self.hermite_interpolation(f)
        f = self.quintic_hermite_interpolation(f)
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])

        return self.mix(w0, w1, f[1])

    def vnoise3(self, p):
        """Args:
            p (numpy.ndarray): 3-dimensional array
        """
        n = np.floor(p)
        v = [self.hash31(n + np.array([i, j, k]))
             for k in range(2) for j in range(2) for i in range(2)]

        f, _ = np.modf(p)
        # f = self.hermite_interpolation(f)
        f = self.quintic_hermite_interpolation(f)
        w0 = self.mix(self.mix(v[0], v[1], f[0]), self.mix(v[2], v[3], f[0]), f[1])
        w1 = self.mix(self.mix(v[4], v[5], f[0]), self.mix(v[6], v[7], f[0]), f[1])

        return self.mix(w0, w1, f[2])

    def vgrad(self, p):
        """Args:
            p (numpy.ndarray): 2-dimensional array
        """
        eps = 0.001
        arr = np.array([
            self.vnoise2(p + np.array([eps, 0.0])) - self.vnoise2(p - np.array([eps, 0.0])),
            self.vnoise2(p + np.array([0.0, eps])) - self.vnoise2(p - np.array([0.0, eps]))
        ])

        grad = 0.5 * arr / eps
        return np.dot(np.ones(2), grad)

    def noise2(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vnoise2(np.array([x, y]) + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise3(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vnoise3(np.array([x + t, y + t, t]))
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def grad2(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vgrad(np.array([x, y]) + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def fractal2(self, size=256, grid=4, t=None, gain=0.5, lacunarity=2.01, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal(self.vnoise2, gain, lacunarity, octaves)

        arr = np.array(
            [noise.fractal(np.array([x, y]) + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def warp2_rot(self, size=256, grid=4, t=None, weight=1, octaves=4):
        t = self.mock_time() if t is None else t
        noise = Fractal(self.vnoise2)
        warp = DomainWarping(noise.fractal, weight, octaves)

        arr = np.array(
            [warp.warp2_rot(np.array([x, y]) + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def warp2(self, size=256, grid=4, octaves=4, t=None):
        t = self.mock_time() if t is None else t
        weight = abs(t % 10 - 5.0)
        noise = Fractal(self.vnoise2)
        warp = DomainWarping(noise.fractal, weight=weight)

        arr = np.array(
            [warp.warp(np.array([x, y]))
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr
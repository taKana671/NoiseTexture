import numpy as np
from pynoise.noise import Noise


class ValueNoise(Noise):

    def vnoise2(self, p):
        n = np.floor(p)
        v = [self.hash21(n + np.array([i, j])) for j in range(2) for i in range(2)]

        f, _ = np.modf(p)
        f = self.fade(f)
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return self.mix(w0, w1, f[1])

    def vnoise3(self, p):
        self.hash = {}
        n = np.floor(p)
        v = [self.hash31(n + np.array([i, j, k]))
             for k in range(2) for j in range(2) for i in range(2)]

        f, _ = np.modf(p)
        f = self.hermite_interpolation(f)
        w0 = self.mix(self.mix(v[0], v[1], f[0]), self.mix(v[2], v[3], f[0]), f[1])
        w1 = self.mix(self.mix(v[4], v[5], f[0]), self.mix(v[6], v[7], f[0]), f[1])
        return self.mix(w0, w1, f[2])

    def noise2(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vnoise2(np.array([x + t, y + t]))
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise3(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.vnoise3(np.array([x + t, y + t, t]))
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    # def wrap2(self, x, y, rot=False):
    #     v = 0.0

    #     for _ in range(4):
    #         cx = np.cos(2 * np.pi * v) if rot else v
    #         sy = np.sin(2 * np.pi * v) if rot else v
    #         x += self.weight * cx
    #         y += self.weight * sy
    #         v = self.fbm2(np.array([x, y]))

    #     return v
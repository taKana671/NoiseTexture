import numpy as np

from .noise import Noise, cache


class PeriodicNoise(Noise):
    """Add periodicity to random numbers to create a periodic noise
       which ends are connnected.
    """

    def __init__(self, period=8.0):
        self.period = period
        self.half_period = self.period * 0.5

    @cache(128)
    def hash11(self, p, loop_n):
        n = p.astype(np.uint32)
        temp_arr = np.zeros(1, dtype=np.uint32)

        for i in range(loop_n):
            temp_arr[0] += n[i]
            self.uhash11(temp_arr)

        return temp_arr[0]

    def gtable2(self, lattice, p):
        idx = self.hash11(lattice, 2)
        idx = idx >> 29

        u = (p[0] if idx < 4 else p[1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[1] if idx < 4 else p[0]) * 0.38268343   # 0.38268343 = sin(pi/8)
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    def gtable3(self, lattice, p):
        idx = self.hash11(lattice, 3)
        idx = idx >> 28
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

    def change_coord(self, x, y):
        px, py = self.xy2pol(2.0 * x - 1, 2.0 * y - 1)

        hx = self.half_period / np.pi * px
        hy = self.half_period * py

        return hx, hy

    def periodic2(self, x, y, t=0.0):
        hx, hy = self.change_coord(x, y)
        p = np.array([hx + t, hy + t])

        n = np.floor(p)
        f = p - n
        # f, _ = np.modf(p)

        v = [self.gtable2(np.mod(n + (arr := np.array([i, j])), self.period), f - arr)
             for j in range(2) for i in range(2)]

        f = self.quintic_hermite_interpolation(f)
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])

        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def periodic3(self, x, y, z):
        hx, hy = self.change_coord(x, y)
        p = np.array([hx, hy, z])
        n = np.floor(p)
        f = p - n
        # f, _ = np.modf(p)

        v = [self.gtable3(np.mod(n + (arr := np.array([i, j, k])), self.period), f - arr) * 0.70710678
             for k in range(2) for j in range(2) for i in range(2)]

        f = self.quintic_hermite_interpolation(f)
        w0 = self.mix(self.mix(v[0], v[1], f[0]), self.mix(v[2], v[3], f[0]), f[1])
        w1 = self.mix(self.mix(v[4], v[5], f[0]), self.mix(v[6], v[7], f[0]), f[1])

        return 0.5 * self.mix(w0, w1, f[2]) + 0.5

    def noise2(self, size=256, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.periodic2(x / size, y / size, t)
                for y in range(size)
                for x in range(size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise3(self, size=256, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.periodic3(x / size, y / size, t)
                for y in range(size)
                for x in range(size)]
        )

        arr = arr.reshape(size, size)
        return arr
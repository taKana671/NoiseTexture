import numpy as np

from noise import Noise


class Periodic(Noise):
    """Add periodicity to random numbers to create a periodic noise
       which ends are connnected.
    """

    def __init__(self, period=4, grid=4, size=256):
        super().__init__(grid, size)
        self.period = period

    def gtable2(self, lattice, p):
        lattice = lattice.astype(np.uint32)

        if (tup := tuple(lattice)) in self.hash:
            idx = self.hash[tup]
        else:
            temp_arr = np.zeros(1, dtype=np.uint32)
            for i in range(2):
                temp_arr[0] += lattice[i]
                self.uhash11(temp_arr)

            idx = temp_arr[0] >> 29
            self.hash[tup] = idx

        u = (p[0] if idx < 4 else p[1]) * 0.92387953   # 0.92387953 = cos(pi/8)
        v = (p[1] if idx < 4 else p[0]) * 0.38268343   # 0.38268343 = sin(pi/8)
        _u = u if idx & 1 == 0 else -u
        _v = v if idx & 2 == 0 else -v

        return _u + _v

    def gtable3(self, lattice, p):
        lattice = lattice.astype(np.uint32)

        if (tup := tuple(lattice)) in self.hash:
            idx = self.hash[tup]
        else:
            temp_arr = np.zeros(1, dtype=np.uint32)
            for i in range(3):
                temp_arr[0] += lattice[i]
                self.uhash11(temp_arr)

            idx = temp_arr[0] >> 28
            self.hash[tup] = idx

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

    def change_coord(self, x, y, t):
        half_grid = self.grid / 2
        half_period = self.period / 2

        px, py = self.xy2pol(x - half_grid, y - half_grid)
        hx = half_period / np.pi * px
        hy = half_period * py

        return hx, hy

    def periodic2(self, x, y, t):
        hx, hy = self.change_coord(x, y, t)
        p = np.array([hx + t, hy + t])
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable2(np.mod(n + (arr := np.array([i, j])), self.period), f - arr)
             for j in range(2) for i in range(2)]

        f = self.fade(f)
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])

        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def periodic3(self, x, y, t):
        hx, hy = self.change_coord(x, y, t)
        p = np.array([hx + t, hy + t, t])
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable3(np.mod(n + (arr := np.array([i, j, k])), self.period), f - arr) * 0.70710678
             for k in range(2) for j in range(2) for i in range(2)]

        f = self.fade(f)
        w0 = self.mix(self.mix(v[0], v[1], f[0]), self.mix(v[2], v[3], f[0]), f[1])
        w1 = self.mix(self.mix(v[4], v[5], f[0]), self.mix(v[6], v[7], f[0]), f[1])

        return 0.5 * self.mix(w0, w1, f[2]) + 0.5

    def noise2(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.periodic2(x, y, t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def noise3(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.periodic3(x, y, t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr
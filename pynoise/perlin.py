import numpy as np

from pynoise.noise import Noise


class Perlin(Noise):

    def __init__(self, weight=0.5, grid=4, size=256):
        self.size = size
        self.grid = grid
        self.weight = weight

    def gtable2(self, lattice, p):
        lattice = lattice.astype(np.uint32)

        if (tup := tuple(lattice)) in self.hash:
            idx = self.hash[tup]
        else:
            self.uhash22(lattice)
            idx = lattice[0] >> 29
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
            self.uhash33(lattice)
            idx = lattice[0] >> 28
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

    def pnoise2(self, p):
        n = np.floor(p)
        f, _ = np.modf(p)

        v = [self.gtable2(n + (arr := np.array([i, j])), f - arr)
             for j in range(2) for i in range(2)]

        f = self.fade(f)
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return 0.5 * self.mix(w0, w1, f[1]) + 0.5

    def pnoise3(self, p):
        n = np.floor(p)
        f, _ = np.modf(p)

        # 0.70710678 = 1 / sqrt(2)
        v = [self.gtable3(n + (arr := np.array([i, j, k])), f - arr) * 0.70710678
             for k in range(2) for j in range(2) for i in range(2)]

        f = self.fade(f)
        w0 = self.mix(self.mix(v[0], v[1], f[0]), self.mix(v[2], v[3], f[0]), f[1])
        w1 = self.mix(self.mix(v[4], v[5], f[0]), self.mix(v[6], v[7], f[0]), f[1])
        return 0.5 * self.mix(w0, w1, f[2]) + 0.5

    def noise2(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.pnoise2(np.array([x + t, y + t]))
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def noise3(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.pnoise3(np.array([x + t, y + t, t]))
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def wrap2(self, x, y, rot=False):
        v = 0.0

        for _ in range(4):
            cx = np.cos(2 * np.pi * v) if rot else v
            sy = np.sin(2 * np.pi * v) if rot else v
            _x = x + self.weight * cx
            _y = y + self.weight * sy
            v = self.pnoise2(np.array([_x, _y]))

        return v
import numpy as np

from .noise import Noise


class VoronoiNoise(Noise):

    def vnoise2(self, p):
        """Args:
            p (numpy.ndarray) 2_dimensional
        """
        n = np.floor(p + 0.5)
        dist = 2.0 ** 0.5
        lattice_pt = np.zeros(2)
        grid = np.zeros(2)

        for j in range(3):
            grid[1] = n[1] + np.sign(j % 2 - .5) * np.ceil(j * .5)

            if abs(grid[1] - p[1]) - 0.5 > dist:
                continue

            for i in range(-1, 2):
                grid[0] = n[0] + i
                jitter = self.hash22(grid) - 0.5

                if (length := self.get_norm(grid + jitter - p)) <= dist:
                    dist = length
                    lattice_pt[:] = grid[:]

        return lattice_pt

    def vnoise3(self, p):
        """Args:
            p (numpy.ndarray) 3_dimensional
        """
        n = np.floor(p + 0.5)
        dist = 3.0 ** 0.5
        lattice_pt = np.zeros(3)
        grid = np.zeros(3)

        for k in range(3):
            grid[2] = n[2] + np.sign(k % 2 - .5) * np.ceil(k * .5)

            if abs(grid[2] - p[2]) - 0.5 > dist:
                continue

            for j in range(3):
                grid[1] = n[1] + np.sign(j % 2 - .5) * np.ceil(j * .5)

                if abs(grid[1] - p[1]) - 0.5 > dist:
                    continue

                for i in range(-1, 2):
                    grid[0] = n[0] + i
                    jitter = self.hash33(grid) - 0.5

                    if (length := self.get_norm(grid + jitter - p)) <= dist:
                        dist = length
                        lattice_pt[:] = grid[:]

        return lattice_pt

    def normalize(self, p):
        if (norm := np.sqrt(np.sum(p**2))) == 0:
            norm = 1

        return p / norm

    def voronoi2(self, x, y):
        """Args:
            x, y (float)
        """
        p = np.array([x, y])
        lattice_pt = self.vnoise2(p)
        return self.hash22(lattice_pt)

    def voronoi_edge2(self, x, y):
        p = np.array([x, y])
        lattice_pt = self.vnoise2(p)

        md = 2.0 ** 0.5
        a = lattice_pt + self.hash22(lattice_pt) - 0.5 - p
        tmp = np.zeros(2)

        for j in range(-2, 3):
            tmp[1] = j

            for i in range(-2, 3):
                tmp[0] = i
                grid = lattice_pt + tmp
                b = grid + self.hash22(grid) - 0.5 - p

                if np.dot(a - b, a - b) > 0.0001:
                    md = min(md, np.dot(0.5 * (a + b), self.normalize(b - a)))

        return md

    def voronoi3(self, x, y, z):
        """Args:
            x, y, z (float)
        """
        p = np.array([x, y, z])
        lattice_pt = self.vnoise3(p)
        return self.hash33(lattice_pt)

    def voronoi_edge3(self, x, y, z):
        p = np.array([x, y, z])
        lattice_pt = self.vnoise3(p)

        md = 3.0 ** 0.5
        a = lattice_pt + self.hash33(lattice_pt) - 0.5 - p
        tmp = np.zeros(3)

        for k in range(-2, 3):
            tmp[2] = k

            for j in range(-2, 3):
                tmp[1] = j

                for i in range(-2, 3):
                    tmp[0] = i
                    grid = lattice_pt + tmp
                    b = grid + self.hash33(grid) - 0.5 - p

                    if np.dot(a - b, a - b) > 0.0001:
                        md = min(md, np.dot(0.5 * (a + b), self.normalize(b - a)))

        return md

    def noise3(self, size=256, grid=4, gray=True, t=None):
        t = self.mock_time() if t is None else t

        if gray:
            vec = np.array([0.3, 0.6, 0.2])
            arr = np.array(
                [np.dot(self.voronoi3(x + t, y + t, t), vec)
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size)
        else:
            arr = np.array(
                [self.voronoi3(x + t, y + t, t)
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size, 3)

        return arr

    def noise2(self, size=256, grid=4, gray=True, t=None):
        t = self.mock_time() if t is None else t

        if gray:
            vec = np.array([0.3, 0.6])

            arr = np.array(
                [np.dot(self.voronoi2(x + t, y + t), vec)
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size)
        else:
            arr = np.array(
                [[*self.voronoi2(x + t, y + t), 1]
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size, 3)

        return arr

    def vmix1(self, v, cell=0.0, edge=1.0):
        return self.mix(edge, cell, self.smoothstep(0.02, 0.04, v))

    def vmix2(self, x, y, edge=1.0):
        edge = np.full(3, edge)
        cell = np.array([*self.voronoi2(x, y), 1.0])
        a = self.smoothstep(0.02, 0.04, self.voronoi_edge2(x, y))

        return self.mix(edge, cell, a)

    def vmix3(self, x, y, z, edge=1.0):
        edge = np.full(3, edge)
        cell = self.voronoi3(x, y, z)
        a = self.smoothstep(0.02, 0.04, self.voronoi_edge3(x, y, z))

        return self.mix(edge, cell, a)

    def edge2(self, size=256, grid=4, bw=True, t=None):
        t = self.mock_time() if t is None else t

        if bw:
            arr = np.array(
                [self.vmix1(self.voronoi_edge2(x + t, y + t))
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size)
        else:
            arr = np.array(
                [self.vmix2(x + t, y + t)
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size, 3)

        return arr

    def edge3(self, size=256, grid=4, bw=True, t=None):
        t = self.mock_time() if t is None else t

        if bw:
            arr = np.array(
                [self.vmix1(self.voronoi_edge3(x + t, y + t, t))
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size)
        else:
            arr = np.array(
                [self.vmix3(x + t, y + t, t)
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size, 3)

        return arr
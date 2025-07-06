import numpy as np

from .voronoi import VoronoiNoise


class VoronoiEdges(VoronoiNoise):

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

    def vmix1(self, v, cell=0.0, edge=1.0):
        return self.mix(edge, cell, self.smoothstep(0.02, 0.04, v))

    def vmix2(self, x, y, cell=1.0, edge=1.0):
        edge = np.full(3, edge)
        cell = np.array([*self.voronoi2(x, y), cell])
        a = self.smoothstep(0.02, 0.04, self.voronoi_edge2(x, y))

        return self.mix(edge, cell, a)

    def vmix3(self, x, y, z, edge=1.0):
        edge = np.full(3, edge)
        cell = self.voronoi3(x, y, z)
        a = self.smoothstep(0.02, 0.04, self.voronoi_edge3(x, y, z))

        return self.mix(edge, cell, a)

    def edge2(self, size=256, grid=4, cell=0.0, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self.voronoi_edge2(x + t, y + t), cell, edge)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def edge2_color(self, size=256, grid=4, cell=1.0, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix2(x + t, y + t, cell, edge)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr

    def edge3(self, size=256, grid=4, cell=0.0, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self.voronoi_edge3(x + t, y + t, t), cell, edge)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def edge3_color(self, size=256, grid=4, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix3(x + t, y + t, t, edge)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr
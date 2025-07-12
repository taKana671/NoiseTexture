import numpy as np

from .voronoi import VoronoiNoise
from .voronoi import TileableVoronoiNoise


class VoronoiEdges(VoronoiNoise):
    """A class to generate voronoi edges.
        Args:
            grid (int): the number of vertical and horizontal grids.
    """

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
                        md = min(md, np.dot(
                            0.5 * (a + b), self.normalize(b - a)))

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

    def noise2(self, size=256, cell=0.0, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self.voronoi_edge2(x + t, y + t), cell, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise2_color(self, size=256, cell=1.0, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix2(x + t, y + t, cell, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr

    def noise3(self, size=256, cell=0.0, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self.voronoi_edge3(x + t, y + t, t), cell, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise3_color(self, size=256, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix3(x + t, y + t, t, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr


class TileableVoronoiEdges(TileableVoronoiNoise, VoronoiEdges):
    """A class to generate tileable voronoi edges.
        Args:
            grid (int): the number of vertical and horizontal grids.
    """

    def vnoise2_edge(self, p, n):
        dist = 2.0 ** 0.5
        grid = np.zeros(2)
        closest_cell = np.zeros(2)

        for j in range(-1, 2):
            grid[1] = j + n[1]

            for i in range(-1, 2):
                grid[0] = i + n[0]
                tiled_cell = self.modulo(grid, self.n_grid)
                to_cell = grid + self.hash22(tiled_cell) - 0.5 - p

                if (length := self.get_norm(to_cell)) <= dist:
                    dist = length
                    closest_cell[:] = to_cell[:]

        return closest_cell

    def voronoi_edge2(self, x, y):
        p = np.array([x, y])
        n = np.floor(p + 0.5)
        closest_cell = self.vnoise2_edge(p, n)

        min_dist = 2.0 ** 0.5
        grid = np.zeros(2)

        for j in range(-2, 3):
            grid[1] = j + n[1]

            for i in range(-2, 3):
                grid[0] = i + n[0]
                tiled_cell = self.modulo(grid, self.n_grid)
                to_cell = grid + self.hash22(tiled_cell) - 0.5 - p

                if self.get_norm(closest_cell - to_cell) > 0.0001:
                    min_dist = min(min_dist, np.dot(
                        0.5 * (closest_cell + to_cell), self.normalize(to_cell - closest_cell)))

        return min_dist

    def voronoi3_edge(self, p, n):
        dist = 3.0 ** 0.5
        grid = np.zeros(3)
        closest_cell = np.zeros(3)

        for k in range(-1, 2):
            grid[2] = k + n[2]

            for j in range(-1, 2):
                grid[1] = j + n[1]

                for i in range(-1, 2):
                    grid[0] = i + n[0]
                    tiled_cell = self.modulo(grid, self.n_grid)
                    to_cell = grid + self.hash33(tiled_cell) - 0.5 - p

                    if (length := self.get_norm(to_cell)) < dist:
                        dist = length
                        closest_cell[:] = to_cell[:]

        return closest_cell

    def voronoi_edge3(self, x, y, z):
        p = np.array([x, y, z])
        n = np.floor(p + 0.5)
        closest_cell = self.voronoi3_edge(p, n)

        min_dist = 3.0 ** 0.5
        grid = np.zeros(3)

        for k in range(-2, 3):
            grid[2] = k + n[2]

            for j in range(-2, 3):
                grid[1] = j + n[1]

                for i in range(-2, 3):
                    grid[0] = i + n[0]
                    tiled_cell = self.modulo(grid, self.n_grid)
                    to_cell = grid + self.hash33(tiled_cell) - 0.5 - p

                    if self.get_norm(closest_cell - to_cell) > 0.0001:
                        min_dist = min(min_dist, np.dot(
                            0.5 * (closest_cell + to_cell), self.normalize(to_cell - closest_cell)))

        return min_dist

import math

import numpy as np

from .edges import VoronoiEdges
from .edges import TileableVoronoiEdges


class VoronoiRoundEdges(VoronoiEdges):
    """A class to generate voronoi round edges.
        Args:
            grid (int): the number of vertical and horizontal grids.
    """

    def min_exp(self, a, b, tp):
        """The smaller the `tp`, the more rounded the voronoi corners.
           tp: how much the tiles are packed.
        """
        res = math.exp(-tp * a) + math.exp(-tp * b)
        return -math.log(res) / tp

    def voronoi_round_edge2(self, x, y, tp=20):
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
                    md = self.min_exp(md, np.dot(
                        0.5 * (a + b), self.normalize(b - a)), tp)

        return md

    def voronoi_round_edge3(self, x, y, z, tp=20):
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
                        md = self.min_exp(md, np.dot(
                            0.5 * (a + b), self.normalize(b - a)), tp)

        return md

    def vmix2_round(self, x, y, cell=1.0, edge=1.0, tp=20):
        edge = np.full(3, edge)
        cell = np.array([*self.voronoi2(x, y), cell])
        a = self.smoothstep(0.02, 0.04, self.voronoi_round_edge2(x, y, tp))

        return self.mix(edge, cell, a)

    def vmix3_round(self, x, y, z, edge=1.0, tp=20):
        edge = np.full(3, edge)
        cell = self.voronoi3(x, y, z)
        a = self.smoothstep(0.02, 0.04, self.voronoi_round_edge3(x, y, z, tp))

        return self.mix(edge, cell, a)

    def noise2(self, size=256, cell=0.0, edge=1.0, tp=20, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self.voronoi_round_edge2(x + t, y + t, tp), cell, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise2_color(self, size=256, cell=1.0, edge=1.0, tp=40, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix2_round(x + t, y + t, cell, edge, tp)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr

    def noise3(self, size=256, cell=0.0, edge=1.0, tp=20, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self.voronoi_round_edge3(x + t, y + t, t, tp), cell, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise3_color(self, size=256, edge=1.0, tp=40, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix3_round(x + t, y + t, t, edge, tp)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr


class TileableVoronoiRoundEdges(TileableVoronoiEdges, VoronoiRoundEdges):
    """A class to generate tileable voronoi round edges.
        Args:
            grid (int): the number of vertical and horizontal grids.
    """

    def voronoi_round_edge2(self, x, y, tp=20):
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
                    min_dist = self.min_exp(min_dist, np.dot(
                        0.5 * (closest_cell + to_cell), self.normalize(to_cell - closest_cell)), tp)

        return min_dist

    def voronoi_round_edge3(self, x, y, z, tp=20):
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
                        min_dist = self.min_exp(min_dist, np.dot(
                            0.5 * (closest_cell + to_cell), self.normalize(to_cell - closest_cell)), tp)

        return min_dist

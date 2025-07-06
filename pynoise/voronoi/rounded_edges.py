import math

import numpy as np

from .edges import VoronoiEdges


class VoronoiRoundEdges(VoronoiEdges):

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
                    md = self.min_exp(md, np.dot(0.5 * (a + b), self.normalize(b - a)), tp)

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
                        md = self.min_exp(md, np.dot(0.5 * (a + b), self.normalize(b - a)), tp)

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

    def round2(self, size=256, grid=4, cell=0.0, edge=1.0, tp=20, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self.voronoi_round_edge2(x + t, y + t, tp), cell, edge)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def round2_color(self, size=256, grid=4, cell=1.0, edge=1.0, tp=40, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix2_round(x + t, y + t, cell, edge, tp)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr

    def round3(self, size=256, grid=4, cell=0.0, edge=1.0, tp=20, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self.voronoi_round_edge3(x + t, y + t, t, tp), cell, edge)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def round3_color(self, size=256, grid=4, edge=1.0, tp=40, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix3_round(x + t, y + t, t, edge, tp)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr

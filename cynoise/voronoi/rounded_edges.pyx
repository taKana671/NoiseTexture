# cython: language_level=3

import cython
import numpy as np
from libc.math cimport floor, ceil, fmin, log, exp


from .edges import VoronoiEdges
# from .voronoi import TileableVoronoiNoise


cdef class VoronoiRoundEdges(VoronoiEdges):

    @cython.cdivision(True)
    cdef double min_exp(self, double a, double b, double tp):
        """The smaller the `tp`, the more rounded the voronoi corners.
           tp: how much the tiles are packed.
        """
        cdef:
            double v
        
        v = exp(-tp * a) + exp(-tp * b)
        return -log(v) / tp
    
    cdef double _voronoi_round_edge2(self, double x, double y, double tp):
        cdef:
            int i, j, ii
            double b
            double[2] grid, lattice_pt, h
            double[2] a, a1, a2, a3, nm
            double[2] p = [x, y]
            double dist = 2.0 ** 0.5

        self._vnoise2(x, y, &lattice_pt)
        self.hash22(&lattice_pt, &h)

        for ii in range(2):
            a[ii] = lattice_pt[ii] + h[ii] - 0.5 - p[ii]

        for j in range(-2, 3):
            grid[1] = lattice_pt[1] + <double>j

            for i in range(-2, 3):
                grid[0] = lattice_pt[0] + <double>i
                self.hash22(&grid, &h)

                for ii in range(2):
                    b = grid[ii] + h[ii] - 0.5 - p[ii]
                    a1[ii] = a[ii] - b
                    a2[ii] = (a[ii] + b) * 0.5
                    a3[ii] = b - a[ii]

                if self.inner_product22(&a1, &a1) > 0.0001:
                    self.normalize2(&a3, &nm)
                    dist = self.min_exp(dist, self.inner_product22(&a2, &nm), tp)

        return dist
    
    cdef double _voronoi_round_edge3(self, double x, double y, double z, double tp):
        cdef:
            int i, j, k, ii
            double b
            double[3] grid, lattice_pt, h
            double[3] a, a1, a2, a3, nm
            double[3] p = [x, y, z]
            double dist = 3.0 ** 0.5

        self._vnoise3(x, y, z, &lattice_pt)
        self.hash33(&lattice_pt, &h)

        for ii in range(3):
            a[ii] = lattice_pt[ii] + h[ii] - 0.5 - p[ii]

        for k in range(-2, 3):
            grid[2] = lattice_pt[2] + <double>k
        
            for j in range(-2, 3):
                grid[1] = lattice_pt[1] + <double>j

                for i in range(-2, 3):
                    grid[0] = lattice_pt[0] + <double>i
                    self.hash33(&grid, &h)

                    for ii in range(3):
                        b = grid[ii] + h[ii] - 0.5 - p[ii]
                        a1[ii] = a[ii] - b
                        a2[ii] = (a[ii] + b) * 0.5
                        a3[ii] = b - a[ii]

                    if self.inner_product33(&a1, &a1) > 0.0001:
                        self.normalize3(&a3, &nm)
                        dist = self.min_exp(dist, self.inner_product33(&a2, &nm), tp)

        return dist

    cpdef double voronoi_round_edge2(self, double x, double y, double tp=20):
        return self._voronoi_round_edge2(x, y, tp)

    cpdef double voronoi_round_edge3(self, double x, double y, double z, double tp=20):
        return self._voronoi_round_edge3(x, y, z, tp)

    cpdef list vmix2_round(self, double x, double y, double cell=1.0, double edge=1.0, double tp=20):
        cdef:
            double[3] edge_cl = [edge, edge, edge]
            double[3] cell_cl, m
            double v1, v2, a

        v1, v2 = self._voronoi2(x, y)
        cell_cl = [v1, v2, cell]
        a = self.smoothstep(0.02, 0.04, self._voronoi_round_edge2(x, y, tp))
        self.mix3(&edge_cl, &cell_cl, a, &m)

        return m

    cpdef list vmix3_round(self, double x, double y, double z, double edge=1.0, double tp=20):
        cdef:
            double[3] edge_cl = [edge, edge, edge]
            double[3] cell_cl, m
            double v1, v2, v3, a

        v1, v2, v3 = self._voronoi3(x, y, z)
        cell_cl = [v1, v2, v3]
        a = self.smoothstep(0.02, 0.04, self._voronoi_round_edge3(x, y, z, tp))
        self.mix3(&edge_cl, &cell_cl, a, &m)

        return m
    
    cpdef round2(self, size=256, cell=0.0, edge=1.0, tp=20, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self._voronoi_round_edge2(x + t, y + t, tp), cell, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    cpdef round2_color(self, size=256, cell=1.0, edge=1.0, tp=40, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix2_round(x + t, y + t, cell, edge, tp)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr

    cpdef round3(self, size=256, self.n_grid, cell=0.0, edge=1.0, tp=20, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self._voronoi_round_edge3(x + t, y + t, t, tp), cell, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    cpdef round3_color(self, size=256, edge=1.0, tp=40, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix3_round(x + t, y + t, t, edge, tp)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr
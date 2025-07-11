# cython: language_level=3

import cython
import numpy as np
from libc.math cimport floor, fmin

from .voronoi cimport VoronoiNoise


cdef class VoronoiEdges(VoronoiNoise):

    cdef double _voronoi_edge2(self, double x, double y):
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
                    dist = fmin(dist, self.inner_product22(&a2, &nm))

        return dist
    
    cdef double _voronoi_edge3(self, double x, double y, double z):
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
                        dist = fmin(dist, self.inner_product33(&a2, &nm))

        return dist
    
    cpdef double voronoi_edge2(self, double x, double y):
        return self._voronoi_edge2(x, y)

    cpdef double voronoi_edge3(self, double x, double y, double z):
        return self._voronoi_edge3(x, y, z)

    cpdef double vmix1(self, double v, double cell=0.0, double edge=1.0):
        return self.mix(edge, cell, self.smoothstep(0.02, 0.04, v))
    
    cpdef list vmix2(self, double x, double y, double cell=1.0, double edge=1.0):
        cdef:
            double[3] edge_cl = [edge, edge, edge]
            double[3] cell_cl, m
            double v1, v2, a

        v1, v2 = self._voronoi2(x, y)
        cell_cl = [v1, v2, cell]
        a = self.smoothstep(0.02, 0.04, self._voronoi_edge2(x, y))
        self.mix3(&edge_cl, &cell_cl, a, &m)

        return m

    cpdef list vmix3(self, double x, double y, double z, double edge=1.0):
        cdef:
            double[3] edge_cl = [edge, edge, edge]
            double[3] cell_cl, m
            double v1, v2, v3, a

        v1, v2, v3 = self._voronoi3(x, y, z)
        cell_cl = [v1, v2, v3]
        a = self.smoothstep(0.02, 0.04, self._voronoi_edge3(x, y, z))
        self.mix3(&edge_cl, &cell_cl, a, &m)

        return m
    
    cpdef noise2(self, size=256, cell=0.0, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self._voronoi_edge2(x + t, y + t), cell, edge)
                for y in np.linspace(0, self.n_grid, size) 
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    cpdef noise2_color(self, size=256, cell=1.0, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix2(x + t, y + t, cell, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)        
        return arr

    cpdef noise3(self, size=256, cell=0.0, edge=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.vmix1(self._voronoi_edge3(x + t, y + t, t), cell, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    cpdef noise3_color(self, size=256, edge=1.0, t=None):
        t = self.mock_time() if t is None else t
    
        arr = np.array(
            [self.vmix3(x + t, y + t, t, edge)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr


cdef class TileableVoronoiEdges(VoronoiEdges):

    cdef void _vnoise2(self, double x, double y, double[2] *lattice_pt):
        cdef:
            int i, j, ii
            double nx, ny, length, v
            double[2] grid, jitter, to_cell, tiled_cell
            double[2] p = [x, y]
            double dist = 2.0 ** 0.5

        nx = floor(p[0] + 0.5)
        ny = floor(p[1] + 0.5)

        for j in range(-1, 2):
            grid[1] = ny + <double>j

            for i in range(-1, 2):
                grid[0] = nx + <double>i
                v = <double>self.n_grid
                self.modulo21(&grid, &v, &tiled_cell)
                self.hash22(&tiled_cell, &jitter)

                for ii in range(2):
                    to_cell[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]

                length = self.get_norm2(&to_cell)

                if length <= dist:
                    dist = length

                    for ii in range(2):
                        lattice_pt[0][ii] = tiled_cell[ii] + jitter[ii]


    cdef void _vnoise2_edge(self, double[2] *p, double[2] *n, double[2] *closest_cell):
        cdef:
            int i, j, ii
            double length, v
            double[2] grid, jitter, to_cell, tiled_cell
            double dist = 2.0 ** 0.5

        for j in range(-1, 2):
            grid[1] = n[0][1] + <double>j

            for i in range(-1, 2):
                grid[0] = n[0][0] + <double>i
                v = <double>self.n_grid
                self.modulo21(&grid, &v, &tiled_cell)
                self.hash22(&tiled_cell, &jitter)

                for ii in range(2):
                    to_cell[ii] = grid[ii] + jitter[ii] - 0.5 - p[0][ii]

                length = self.get_norm2(&to_cell)

                if length <= dist:
                    dist = length
                    closest_cell[0] = to_cell

    cdef double _voronoi_edge2(self, double x, double y):
        cdef:
            int i, j, ii
            double v
            double[2] grid, closest_cell, tiled_cell, jitter, to_cell
            double[2] n, diff, a, b, nm
            double[2] p = [x, y]
            double min_dist = 2.0 ** 0.5

        n[0] = floor(p[0] + 0.5)
        n[1] = floor(p[1] + 0.5)
        self._vnoise2_edge(&p, &n, &closest_cell)

        for j in range(-2, 3):
            grid[1] = n[1] + <double>j

            for i in range(-2, 3):
                grid[0] = n[0] + <double>i
                v = <double>self.n_grid
                self.modulo21(&grid, &v, &tiled_cell)
                self.hash22(&tiled_cell, &jitter)

                for ii in range(2):
                    to_cell[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]
                    diff[ii] = closest_cell[ii] - to_cell[ii]

                if self.get_norm2(&diff) > 0.0001:
                    for ii in range(2):
                        a[ii] = (closest_cell[ii] + to_cell[ii]) * 0.5
                        b[ii] = to_cell[ii] - closest_cell[ii]

                    self.normalize2(&b, &nm)
                    min_dist = fmin(min_dist, self.inner_product22(&a, &nm))

        return min_dist
    
    cdef void _vnoise3(self, double x, double y, double z, double[3] *lattice_pt):
        cdef:
            int i, j, k, ii
            double nx, ny, nz, length, v
            double[3] grid, jitter, tiled_cell, to_cell
            double[3] p = [x, y, z]
            double dist = 3.0 ** 0.5

        nx = floor(p[0] + 0.5)
        ny = floor(p[1] + 0.5)
        nz = floor(p[2] + 0.5)

        for k in range(-1, 2):
            grid[2] = nz + <double>k

            for j in range(-1, 2):
                grid[1] = ny + <double>j

                for i in range(-1, 2):
                    grid[0] = nx + <double>i
                    v = <double>self.n_grid
                    self.modulo31(&grid, &v, &tiled_cell)
                    self.hash33(&tiled_cell, &jitter)

                    for ii in range(3):
                        to_cell[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]

                    length = self.get_norm3(&to_cell)

                    if length <= dist:
                        dist = length

                        for ii in range(3):
                            lattice_pt[0][ii] = tiled_cell[ii] + jitter[ii]
    
    cdef void _vnoise3_edge(self, double[3] *p, double[3] *n, double[3] *closest_cell):
        cdef:
            int i, j, k, ii
            double length, v
            double[3] grid, jitter, to_cell, tiled_cell
            double dist = 3.0 ** 0.5

        for k in range(-1, 2):
            grid[2] = n[0][2] + <double>k

            for j in range(-1, 2):
                grid[1] = n[0][1] + <double>j

                for i in range(-1, 2):
                    grid[0] = n[0][0] + <double>i
                    v = <double>self.n_grid
                    self.modulo31(&grid, &v, &tiled_cell)
                    self.hash33(&tiled_cell, &jitter)

                    for ii in range(3):
                        to_cell[ii] = grid[ii] + jitter[ii] - 0.5 - p[0][ii]

                    length = self.get_norm3(&to_cell)

                    if length <= dist:
                        dist = length
                        closest_cell[0] = to_cell

    cdef double _voronoi_edge3(self, double x, double y, double z):
        cdef:
            int i, j, k, ii
            double v
            double[3] grid, closest_cell, tiled_cell, jitter, to_cell
            double[3] n, diff, a, b, nm
            double[3] p = [x, y, z]
            double min_dist = 3.0 ** 0.5

        for ii in range(3):
            n[ii] = floor(p[ii] + 0.5)        
        
        self._vnoise3_edge(&p, &n, &closest_cell)

        for k in range(-2, 3):
            grid[2] = n[2] + <double>k

            for j in range(-2, 3):
                grid[1] = n[1] + <double>j

                for i in range(-2, 3):
                    grid[0] = n[0] + <double>i
                    v = <double>self.n_grid
                    self.modulo31(&grid, &v, &tiled_cell)
                    self.hash33(&tiled_cell, &jitter)

                    for ii in range(3):
                        to_cell[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]
                        diff[ii] = closest_cell[ii] - to_cell[ii]

                    if self.get_norm3(&diff) > 0.0001:
                        for ii in range(3):
                            a[ii] = (closest_cell[ii] + to_cell[ii]) * 0.5
                            b[ii] = to_cell[ii] - closest_cell[ii]

                        self.normalize3(&b, &nm)
                        min_dist = fmin(min_dist, self.inner_product33(&a, &nm))

        return min_dist

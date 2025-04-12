# cython: language_level=3

import numpy as np
from libc.math cimport floor, ceil, fmin

from .noise cimport Noise


cdef class VoronoiNoise(Noise):

    cdef void _vnoise2(self, double x, double y, double[2] *lattice_pt):
        cdef:
            int i, j, ii
            double nx, ny, length, md
            double[2] grid, jitter, v
            double[2] p = [x, y]
            double dist = 2.0 ** 0.5

        nx = floor(p[0] + 0.5)
        ny = floor(p[1] + 0.5)

        for j in range(3):
            md = self.mod(<double>j, 2.0) - 0.5
            grid[1] = ny + self.sign_with_abs(&md) * ceil(j * 0.5)

            if abs(grid[1] - p[1]) - 0.5 > dist:
                continue

            for i in range(-1, 2):
                grid[0] = nx + <double>i
                self.hash22(&grid, &jitter)

                for ii in range(2):
                    v[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]

                length = self.get_norm2(&v)

                if length <= dist:
                    dist = length
                    lattice_pt[0] = grid

    cdef list _voronoi2(self, double x, double y):
        cdef:
            double[2] lattice_pt
            double[2] h

        self._vnoise2(x, y, &lattice_pt)
        self.hash22(&lattice_pt, &h)

        return h

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

    cdef void _vnoise3(self, double x, double y, double z, double[3] *lattice_pt):
        cdef:
            int i, j, k, ii
            double nx, ny, nz, length, md
            double[3] grid, jitter, v
            double[3] p = [x, y, z]
            double dist = 3.0 ** 0.5

        nx = floor(p[0] + 0.5)
        ny = floor(p[1] + 0.5)
        nz = floor(p[2] + 0.5)

        for k in range(3):
            md = self.mod(<double>k, 2.0) - 0.5
            grid[2] = nz + self.sign_with_abs(&md) * ceil(k * 0.5)

            if abs(grid[2] - p[2]) - 0.5 > dist:
                continue

            for j in range(3):
                md = self.mod(<double>j, 2.0) - 0.5
                grid[1] = ny + self.sign_with_abs(&md) * ceil(j * 0.5)

                if abs(grid[1] - p[1]) - 0.5 > dist:
                    continue

                for i in range(-1, 2):
                    grid[0] = nx + i
                    self.hash33(&grid, &jitter)

                    for ii in range(3):
                        v[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]                    

                    length = self.get_norm3(&v)
                    if length <= dist:
                        dist = length
                        lattice_pt[0] = grid
    
    cdef list _voronoi3(self, double x, double y, double z):
        cdef:
            double[3] lattice_pt
            double[3] h

        self._vnoise3(x, y, z, &lattice_pt)
        self.hash33(&lattice_pt, &h)

        return h

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

    cpdef list voronoi2(self, double x, double y):
        return self._voronoi2(x, y)

    cpdef list voronoi3(self, double x, double y, double z):
        return self._voronoi3(x, y, z)

    cpdef double voronoi_edge2(self, double x, double y):
        return self._voronoi_edge2(x, y)

    cpdef double voronoi_edge3(self, double x, double y, double z):
        return self._voronoi_edge3(x, y, z)
    
    cpdef noise2(self, size=256, grid=4, gray=True, t=None):
        t = self.mock_time() if t is None else float(t)

        if gray:
            vec = [0.3, 0.6]
            arr = np.array(
                [np.dot(self._voronoi2(x + t, y + t), vec)
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size)
        else:
            arr = np.array(
                [[*self._voronoi2(x + t, y + t), 1]
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size, 3)

        return arr

    cpdef noise3(self, size=256, grid=4, gray=True, t=None):
        t = self.mock_time() if t is None else float(t)

        if gray:
            vec = [0.3, 0.6, 0.2]
            arr = np.array(
                [np.dot(self._voronoi3(x + t, y + t, t), vec)
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size)
        else:
            arr = np.array(
                [self._voronoi3(x + t, y + t, t)
                    for y in np.linspace(0, grid, size)
                    for x in np.linspace(0, grid, size)]
            )
            arr = arr.reshape(size, size, 3)

        return arr

    cpdef double vmix1(self, double v, double cell=0.0, double edge=1.0):
        return self.mix(edge, cell, self.smoothstep(0.02, 0.04, v))
    
    cpdef list vmix2(self, double x, double y, double edge=1.0):
        cdef:
            double[3] edge_cl = [edge, edge, edge]
            double[3] cell_cl, m
            double v1, v2, a

        v1, v2 = self._voronoi2(x, y)
        cell_cl = [v1, v2, 1.0]
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
    
    cpdef edge2(self, size=256, grid=4, bw=True, t=None):
        t = self.mock_time() if t is None else t

        if bw:
            arr = np.array(
                [self.vmix1(self._voronoi_edge2(x + t, y + t))
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

    cpdef edge3(self, size=256, grid=4, bw=True, t=None):
        t = self.mock_time() if t is None else t

        if bw:
            arr = np.array(
                [self.vmix1(self._voronoi_edge3(x + t, y + t, t))
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
    
    

        

        










        
import cython
import numpy as np
from libc.math cimport floor, ceil

from ..noise cimport Noise


cdef class VoronoiNoise(Noise):

    def __init__(self, grid=4):
        super().__init__()
        self.n_grid = grid

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

    cdef list _voronoi2(self, double x, double y):
        cdef:
            double[2] lattice_pt
            double[2] h

        self._vnoise2(x, y, &lattice_pt)
        self.hash22(&lattice_pt, &h)

        return h

    cdef list _voronoi3(self, double x, double y, double z):
        cdef:
            double[3] lattice_pt
            double[3] h

        self._vnoise3(x, y, z, &lattice_pt)
        self.hash33(&lattice_pt, &h)

        return h

    cpdef list voronoi2(self, double x, double y):
        return self._voronoi2(x, y)

    cpdef list voronoi3(self, double x, double y, double z):
        return self._voronoi3(x, y, z)

    cpdef noise2(self, size=256, t=None):
        t = self.mock_time() if t is None else float(t)
        vec = [0.3, 0.6]

        arr = np.array(
            [np.dot(self._voronoi2(x + t, y + t), vec)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    cpdef noise2_color(self, size=256, cell=1.0, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [[*self._voronoi2(x + t, y + t), cell]
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr

    cpdef noise3(self, size=256, t=None):
        t = self.mock_time() if t is None else float(t)
        vec = [0.3, 0.6, 0.2]

        arr = np.array(
            [np.dot(self._voronoi3(x + t, y + t, t), vec)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    cpdef noise3_color(self, size=256, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self._voronoi3(x + t, y + t, t)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr


cdef class TileableVoronoiNoise(VoronoiNoise):

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

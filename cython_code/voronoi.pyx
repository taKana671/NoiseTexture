import numpy as np
from libc.math cimport floor

from noise cimport Noise


cdef class Voronoi(Noise):

    def __init__(self, grid=4, size=256):
        super().__init__(grid, size)

    cdef list voronoi2(self, double x, double y):
        cdef:
            int i, j, ii
            double nx, ny
            double[2] grid, jitter, lattice_pt, v, h
            double[2] p = [x, y]
            double dist = 2.0 ** 0.5

        nx = floor(p[0] + 0.5)
        ny = floor(p[1] + 0.5)

        for j in (0, 1, -1):
            grid[1] = ny + j
            if abs(grid[1] - p[1]) - 0.5 > dist:
                continue

            for i in range(-1, 2):
                grid[0] = nx + i
                self.hash22(&grid, &jitter)

                for ii in range(2):
                    v[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]
                
                length = self.get_norm2(&v)
                if length <= dist:
                    dist = length
                    lattice_pt = grid

        self.hash22(&lattice_pt, &h)
        return h

    cdef list voronoi3(self, double x, double y, double z):
        cdef:
            int i, j, k, ii
            double nx, ny, nz
            double[3] grid, jitter, lattice_pt, v, h
            double[3] p = [x, y, z]
            double dist = 3.0 ** 0.5

        nx = floor(p[0] + 0.5)
        ny = floor(p[1] + 0.5)
        nz = floor(p[2] + 0.5)

        for k in (0, 1, -1):
            grid[2] = nz + k
            if abs(grid[2] - p[2]) - 0.5 > dist:
                continue

            for j in (0, 1, -1):
                grid[1] = ny + j
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
                        lattice_pt = grid

        self.hash33(&lattice_pt, &h)
        return h

    cpdef noise3(self, gray=True):
        t = self.mock_time()

        if gray:
            vec = [0.3, 0.6, 0.2]
            arr = np.array(
                [np.dot(self.voronoi3(x + t, y + t, t), vec)
                    for y in np.linspace(0, self.grid, self.size)
                    for x in np.linspace(0, self.grid, self.size)]
            )
            arr = arr.reshape(self.size, self.size)
        else:
            arr = np.array(
                [self.voronoi3(x + t, y + t, t)
                    for y in np.linspace(0, self.grid, self.size)
                    for x in np.linspace(0, self.grid, self.size)]
            )
            arr = arr.reshape(self.size, self.size, 3)

        return arr

    cpdef noise2(self, gray=True):
        t = self.mock_time()

        if gray:
            vec = [0.3, 0.6]
            arr = np.array(
                [np.dot(self.voronoi2(x + t, y + t), vec)
                    for y in np.linspace(0, self.grid, self.size)
                    for x in np.linspace(0, self.grid, self.size)]
            )
            arr = arr.reshape(self.size, self.size)
        else:
            arr = np.array(
                [[*self.voronoi2(x + t, y + t), 1]
                    for y in np.linspace(0, self.grid, self.size)
                    for x in np.linspace(0, self.grid, self.size)]
            )
            arr = arr.reshape(self.size, self.size, 3)

        return arr
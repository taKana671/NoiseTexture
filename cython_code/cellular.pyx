import numpy as np
from libc.math cimport floor

from noise cimport Noise


cdef class Cellular(Noise):

    def __init__(self, grid=4, size=256):
        super().__init__(grid, size)

    cdef void sort4(self, double[4] *dist4, double *length):
        if self.step(length[0], dist4[0][0]):
            dist4[0][0], dist4[0][1], dist4[0][2], dist4[0][3] = length[0], dist4[0][0], dist4[0][1], dist4[0][2]

        elif self.step(length[0], dist4[0][1]):
            dist4[0][0], dist4[0][1], dist4[0][2], dist4[0][3] = dist4[0][0], length[0], dist4[0][1], dist4[0][2]

        elif self.step(length[0], dist4[0][2]):
            dist4[0][0], dist4[0][1], dist4[0][2], dist4[0][3] = dist4[0][0], dist4[0][1], length[0], dist4[0][2]

        elif self.step(length[0], dist4[0][3]):
            dist4[0][0], dist4[0][1], dist4[0][2], dist4[0][3] = dist4[0][0], dist4[0][1], dist4[0][2], length[0]
    
    cdef list fdist24(self, double x, double y):
        cdef:
            int i, j, ii
            double nx, ny
            double[2] grid, jitter, v, temp
            double[2] p = [x, y]
            double[4] dist4

        nx = floor(p[0] + 0.5)
        ny = floor(p[1] + 0.5)
        temp[0] = 1.5 - abs(x - nx)
        temp[1] = 1.5 - abs(y - ny)
        length = self.get_norm2(&temp)

        for i in range(4):
            dist4[0] = length

        for j in (0, 1, -1, 2, -2):
            grid[1] = ny + j
            if abs(grid[1] - p[1]) - 0.5 > dist4[3]:
                continue

            for i in range(-2, 3):
                grid[0] = nx + i
                self.hash22(&grid, &jitter)

                for ii in range(2):
                    v[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]

                length = self.get_norm2(&v)
                self.sort4(&dist4, &length)

        return dist4

    cdef double fdist2(self, double x, double y):
        cdef:
            int i, j, ii
            double nx, ny
            double[2] grid, jitter, v
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
                dist = min(dist, length)

        return dist

    cdef double fdist3(self, double x, double y, double z):
        cdef:
            int i, j, k, ii
            double nx, ny, nz
            double[3] grid, jitter, v
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
                    dist = min(dist, length)

        return dist

    cpdef noise2(self):
        t = self.mock_time()

        arr = np.array(
            [self.fdist2(x + t, y + t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    cpdef noise3(self):
        t = self.mock_time()

        arr = np.array(
            [self.fdist3(x + t, y + t, t)
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    cpdef noise24(self, nearest=2):
        t = self.mock_time()

        arr = np.array(
            [self.fdist24(x + t, y + t)[nearest - 1]
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr
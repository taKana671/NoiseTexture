# cython: language_level=3

import numpy as np
from libc.math cimport floor, ceil

from cynoise.noise cimport Noise


cdef class CellularNoise(Noise):

    cdef void _sort4(self, double[4] *dist4, double *length):
        if self.step(length[0], dist4[0][0]):
            dist4[0][0], dist4[0][1], dist4[0][2], dist4[0][3] = length[0], dist4[0][0], dist4[0][1], dist4[0][2]

        elif self.step(length[0], dist4[0][1]):
            dist4[0][0], dist4[0][1], dist4[0][2], dist4[0][3] = dist4[0][0], length[0], dist4[0][1], dist4[0][2]

        elif self.step(length[0], dist4[0][2]):
            dist4[0][0], dist4[0][1], dist4[0][2], dist4[0][3] = dist4[0][0], dist4[0][1], length[0], dist4[0][2]

        elif self.step(length[0], dist4[0][3]):
            dist4[0][0], dist4[0][1], dist4[0][2], dist4[0][3] = dist4[0][0], dist4[0][1], dist4[0][2], length[0]

    #cdef list _fdist24(self, double x, double y):
    cdef (double, double, double, double) _fdist24(self, double x, double y):
        cdef:
            unsigned int i, j, ii
            double nx, ny, length, md
            double[2] grid, jitter, v, temp
            double[2] p = [x, y]
            double[4] dist4

        nx = floor(p[0] + 0.5)
        ny = floor(p[1] + 0.5)
        temp[0] = 1.5 - abs(x - nx)
        temp[1] = 1.5 - abs(y - ny)
        length = self.get_norm2(&temp)

        for i in range(4):
            dist4[i] = length

        for j in range(5):
            md = self.mod(<double>j, 2.0) - 0.5
            grid[1] = ny + self.sign_with_abs(&md) * ceil(j * 0.5)

            if abs(grid[1] - p[1]) - 0.5 > dist4[3]:
                continue

            for i in range(-2, 3):
                grid[0] = nx + i
                self.hash22(&grid, &jitter)

                for ii in range(2):
                    v[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]

                length = self.get_norm2(&v)
                self._sort4(&dist4, &length)

        return dist4[0], dist4[1], dist4[2], dist4[3]
        #return dist4
    
    #cdef list _fdist34(self, double x, double y, double z):
    cdef (double, double, double, double) _fdist34(self, double x, double y, double z):
        cdef:
            unsigned int i, j, k, ii
            double nx, ny, nz, length, md
            double[3] grid, jitter, v, temp
            double[3] p = [x, y, z]
            double[4] dist4

        nx = floor(p[0] + 0.5)
        ny = floor(p[1] + 0.5)
        nz = floor(p[2] + 0.5)
        temp[0] = 1.5 - abs(x - nx)
        temp[1] = 1.5 - abs(y - ny)
        temp[2] = 1.5 - abs(z - nz)
        length = self.get_norm3(&temp)

        for i in range(4):
            dist4[i] = length

        for k in range(5):
            md = self.mod(<double>k, 2.0) - 0.5
            grid[2] = nz + self.sign_with_abs(&md) * ceil(k * 0.5)

            if abs(grid[2] - p[2]) - 0.5 > dist4[3]:
                continue

            for j in range(5):
                md = self.mod(<double>j, 2.0) - 0.5
                grid[1] = ny + self.sign_with_abs(&md) * ceil(j * 0.5)

                if abs(grid[1] - p[1]) - 0.5 > dist4[3]:
                    continue

                for i in range(-2, 3):
                    grid[0] = nx + i
                    self.hash33(&grid, &jitter)

                    for ii in range(3):
                        v[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]

                    length = self.get_norm3(&v)
                    self._sort4(&dist4, &length)

        return dist4[0], dist4[1], dist4[2], dist4[3]
        #return dist4

    cdef double _fdist2(self, double x, double y):
        cdef:
            unsigned int i, j, ii
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
                grid[0] = nx + i
                self.hash22(&grid, &jitter)

                for ii in range(2):
                    v[ii] = grid[ii] + jitter[ii] - 0.5 - p[ii]

                length = self.get_norm2(&v)
                dist = min(dist, length)

        return dist

    cdef double _fdist3(self, double x, double y, double z):
        cdef:
            unsigned int i, j, k, ii
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
                    dist = min(dist, length)

        return dist

    cpdef double fdist2(self, double x, double y):
        return self._fdist2(x, y)

    cpdef double fdist3(self, double x, double y, double z):
        return self._fdist3(x, y, z)

    cpdef list fdist24(self, double x, double y):
        return self._fdist24(x, y)

    cpdef list fdist34(self, double x, double y, double z):
        return self._fdist34(x, y, z)

    cpdef noise2(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self._fdist2(x + t, y + t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    cpdef noise3(self, size=256, grid=4, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self._fdist3(x + t, y + t, t)
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    cpdef noise24(self, size=256, grid=4, nearest=2, t=None):
        t = self.mock_time() if t is None else float(t)

        arr = np.array(
            [self._fdist24(x + t, y + t)[nearest - 1]
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    cpdef cnoise2(self, size=256, grid=4, wx=0.5, wy=-1.0, wz=1.4, ww=-0.1, t=None):
        t = self.mock_time() if t is None else float(t)
        wt = np.array([wx, wy, wz, ww])

        arr = np.array(
            [abs(np.dot(wt, self._fdist24(x + t, y + t)))
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr
    
    cpdef cnoise3(self, size=256, grid=4, wx=0.5, wy=-1.0, wz=1.4, ww=-0.1, t=None):
        t = self.mock_time() if t is None else float(t)
        wt = np.array([wx, wy, wz, ww])

        arr = np.array(
            [abs(np.dot(wt, self._fdist34(x + t, y + t, t)))
                for y in np.linspace(0, grid, size)
                for x in np.linspace(0, grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr
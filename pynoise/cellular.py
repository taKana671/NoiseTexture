import numpy as np

from .noise import Noise


class CellularNoise(Noise):

    def __init__(self, grid=4):
        self.n_grid = grid

    def sort4(self, dist4, length):

        if self.step(length, dist4[0]):
            return np.array([length, *dist4[:3]])

        if self.step(length, dist4[1]):
            return np.array([dist4[0], length, *dist4[1:3]])

        if self.step(length, dist4[2]):
            return np.array([*dist4[:2], length, dist4[2]])

        if self.step(length, dist4[3]):
            return np.array([*dist4[:3], length])

    def fdist24(self, x, y):
        """Compute the 1st, 2nd, 3rd and 4th 2D nearest neighbor distance.
            Args:
                px, py (float)
        """
        p = np.array([x, y])
        n = np.floor(p + 0.5)
        temp = 1.5 - np.abs(p - n)
        length = sum(v ** 2 for v in temp) ** 0.5
        dist4 = np.full(4, length)
        grid = np.zeros(2)

        for j in range(5):
            grid[1] = n[1] + np.sign(j % 2 - .5) * np.ceil(j * .5)
            if abs(grid[1] - p[1]) - 0.5 > dist4[3]:
                continue

            for i in range(-2, 3):
                grid[0] = n[0] + i
                jitter = self.hash22(grid) - 0.5
                length = self.get_norm(grid + jitter - p)

                if (sorted_dist4 := self.sort4(dist4, length)) is not None:
                    dist4 = sorted_dist4

        return dist4

    def fdist34(self, x, y, z):
        """Compute the 1st, 2nd, 3rd and 4th 3D nearest neighbor distance.
            Args:
                px, py, pz (float)
        """
        p = np.array([x, y, z])
        n = np.floor(p + 0.5)
        temp = 1.5 - np.abs(p - n)
        length = sum(v ** 2 for v in temp) ** 0.5
        dist4 = np.full(4, length)
        grid = np.zeros(3)

        for k in range(5):
            grid[2] = n[2] + np.sign(k % 2 - .5) * np.ceil(k * .5)
            if abs(grid[2] - p[2]) - 0.5 > dist4[3]:
                continue

            for j in range(5):
                grid[1] = n[1] + np.sign(j % 2 - .5) * np.ceil(j * .5)
                if abs(grid[1] - p[1]) - 0.5 > dist4[3]:
                    continue

                for i in range(-2, 3):
                    grid[0] = n[0] + i
                    jitter = self.hash33(grid) - 0.5
                    length = self.get_norm(grid + jitter - p)

                    if (sorted_dist4 := self.sort4(dist4, length)) is not None:
                        dist4 = sorted_dist4

        return dist4

    def fdist2(self, x, y):
        """Compute 2D nearest neighbor distance.
            Args:
                px, py (float)
        """
        p = np.array([x, y])
        n = np.floor(p + 0.5)
        dist = 2.0 ** 0.5
        grid = np.zeros(2)

        for j in range(3):
            grid[1] = n[1] + np.sign(j % 2 - .5) * np.ceil(j * .5)
            if abs(grid[1] - p[1]) - 0.5 > dist:
                continue

            for i in range(-1, 2):
                grid[0] = n[0] + i
                jitter = self.hash22(grid) - 0.5
                length = self.get_norm(grid + jitter - p)
                dist = min(dist, length)

        return dist

    def fdist3(self, x, y, z):
        """Compute 3D nearest neighbor distance.
            Args:
                px, py, pz (float)
        """
        p = np.array([x, y, z])
        n = np.floor(p + 0.5)
        dist = 3.0 ** 0.5
        grid = np.zeros(3)

        for k in range(3):
            grid[2] = n[2] + np.sign(k % 2 - .5) * np.ceil(k * .5)
            if abs(grid[2] - p[2]) - 0.5 > dist:
                continue

            for j in range(3):
                grid[1] = n[1] + np.sign(j % 2 - .5) * np.ceil(j * .5)
                if abs(grid[1] - p[1]) - 0.5 > dist:
                    continue

                for i in range(-1, 2):
                    grid[0] = n[0] + i
                    jitter = self.hash33(grid) - 0.5
                    length = self.get_norm(grid + jitter - p)
                    dist = min(dist, length)

        return dist

    def noise2(self, size=256, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.fdist2(x + t, y + t)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def noise3(self, size=256, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.fdist3(x + t, y + t, t)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def noise24(self, size=256, nearest=2, t=None):
        """Return numpy.ndarray to be convert an image.
            Args:
                nearest (int):
                    the order of nearest neighbor distance;
                    must be from 1 to 4.
        """
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.fdist24(x + t, y + t)[nearest - 1]
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def cnoise2(self, size=256, wx=0.5, wy=-1.0, wz=1.4, ww=-0.1, t=None):
        t = self.mock_time() if t is None else t
        wt = np.array([wx, wy, wz, ww])

        arr = np.array(
            [abs(np.dot(wt, self.fdist24(x + t, y + t)))
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr

    def cnoise3(self, size=256, wx=0.5, wy=-1.0, wz=1.4, ww=-0.1, t=None):
        t = self.mock_time() if t is None else t
        wt = np.array([wx, wy, wz, ww])

        arr = np.array(
            [abs(np.dot(wt, self.fdist34(x + t, y + t, t)))
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )
        arr = arr.reshape(size, size)
        return arr


class TileableCellularNoise(CellularNoise):

    def fdist2(self, x, y):
        """Compute 2D nearest neighbor distance.
            Args:
                px, py (float)
        """
        p = np.array([x, y])
        n = np.floor(p + 0.5)
        dist = 2.0 ** 0.5
        grid = np.zeros(2)

        for j in range(-1, 2):
            grid[1] = n[1] + j

            for i in range(-1, 2):
                grid[0] = n[0] + i
                tiled_cell = self.modulo(grid, self.n_grid)
                jitter = self.hash22(tiled_cell)
                length = self.get_norm(grid + jitter - p - 0.5)
                dist = min(dist, length)

        return dist

    def fdist3(self, x, y, z):
        """Compute 3D nearest neighbor distance.
            Args:
                px, py, pz (float)
        """
        p = np.array([x, y, z])
        n = np.floor(p + 0.5)
        dist = 3.0 ** 0.5
        grid = np.zeros(3)

        for k in range(-1, 2):
            grid[2] = k + n[2]

            for j in range(-1, 2):
                grid[1] = j + n[1]

                for i in range(-1, 2):
                    grid[0] = i + n[0]
                    tiled_cell = self.modulo(grid, self.n_grid)
                    jitter = self.hash33(tiled_cell)
                    length = self.get_norm(grid + jitter - p - 0.5)
                    dist = min(dist, length)

        return dist

    def fdist24(self, x, y):
        """Compute the 1st, 2nd, 3rd and 4th 2D nearest neighbor distance.
            Args:
                px, py (float)
        """
        p = np.array([x, y])
        n = np.floor(p + 0.5)
        temp = 1.5 - np.abs(p - n)
        length = sum(v ** 2 for v in temp) ** 0.5
        dist4 = np.full(4, length)
        grid = np.zeros(2)

        for j in range(-2, 3):
            grid[1] = n[1] + j

            for i in range(-2, 3):
                grid[0] = n[0] + i
                tiled_cell = self.modulo(grid, self.n_grid)
                jitter = self.hash22(tiled_cell)
                length = self.get_norm(grid + jitter - p - 0.5)

                if (sorted_dist4 := self.sort4(dist4, length)) is not None:
                    dist4 = sorted_dist4

        return dist4

    def fdist34(self, x, y, z):
        """Compute the 1st, 2nd, 3rd and 4th 3D nearest neighbor distance.
            Args:
                px, py, pz (float)
        """
        p = np.array([x, y, z])
        n = np.floor(p + 0.5)
        temp = 1.5 - np.abs(p - n)
        length = sum(v ** 2 for v in temp) ** 0.5
        dist4 = np.full(4, length)
        grid = np.zeros(3)

        for k in range(-2, 3):
            grid[2] = k + n[2]

            for j in range(-2, 3):
                grid[1] = j + n[1]

                for i in range(-2, 3):
                    grid[0] = i + n[0]

                    tiled_cell = self.modulo(grid, self.n_grid)
                    jitter = self.hash33(tiled_cell)
                    length = self.get_norm(grid + jitter - p - 0.5)

                    if (sorted_dist4 := self.sort4(dist4, length)) is not None:
                        dist4 = sorted_dist4

        return dist4
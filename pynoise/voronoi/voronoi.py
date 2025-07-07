import numpy as np

from ..noise import Noise


class VoronoiNoise(Noise):
    """A class to generate voronoi noise.
        Args:
            grid (int): the number of vertical and horizontal grids.
    """

    def __init__(self, grid=4):
        self.n_grid = grid

    def vnoise2(self, p):
        """Args:
            p (numpy.ndarray) 2-element array
        """
        n = np.floor(p + 0.5)
        dist = 2.0 ** 0.5
        lattice_pt = np.zeros(2)
        grid = np.zeros(2)

        for j in range(3):
            grid[1] = n[1] + np.sign(j % 2 - .5) * np.ceil(j * .5)

            if abs(grid[1] - p[1]) - 0.5 > dist:
                continue

            for i in range(-1, 2):
                grid[0] = n[0] + i
                jitter = self.hash22(grid) - 0.5

                if (length := self.get_norm(grid + jitter - p)) <= dist:
                    dist = length
                    lattice_pt[:] = grid[:]

        return lattice_pt

    def vnoise3(self, p):
        """Args:
            p (numpy.ndarray) 3-element array
        """
        n = np.floor(p + 0.5)
        dist = 3.0 ** 0.5
        lattice_pt = np.zeros(3)
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

                    if (length := self.get_norm(grid + jitter - p)) <= dist:
                        dist = length
                        lattice_pt[:] = grid[:]

        return lattice_pt

    def voronoi2(self, x, y):
        """Args:
            x, y (float)
        """
        p = np.array([x, y])
        lattice_pt = self.vnoise2(p)
        return self.hash22(lattice_pt)

    def voronoi3(self, x, y, z):
        """Args:
            x, y, z (float)
        """
        p = np.array([x, y, z])
        lattice_pt = self.vnoise3(p)
        return self.hash33(lattice_pt)

    def noise3(self, size=256, t=None):
        t = self.mock_time() if t is None else t
        vec = np.array([0.3, 0.6, 0.2])

        arr = np.array(
            [np.dot(self.voronoi3(x + t, y + t, t), vec)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise3_color(self, size=256, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [self.voronoi3(x + t, y + t, t)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr

    def noise2(self, size=256, t=None):
        t = self.mock_time() if t is None else t
        vec = np.array([0.3, 0.6])

        arr = np.array(
            [np.dot(self.voronoi2(x + t, y + t), vec)
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size)
        return arr

    def noise2_color(self, size=256, cell=1.0, t=None):
        t = self.mock_time() if t is None else t

        arr = np.array(
            [[*self.voronoi2(x + t, y + t), cell]
                for y in np.linspace(0, self.n_grid, size)
                for x in np.linspace(0, self.n_grid, size)]
        )

        arr = arr.reshape(size, size, 3)
        return arr


class TileableVoronoiNoise(VoronoiNoise):
    """A class to generate tileable voronoi noise.
        Args:
            grid (int): the number of vertical and horizontal grids.
    """

    def vnoise2(self, p):
        """Args:
            p (numpy.ndarray) 2-element array
        """
        n = np.floor(p + 0.5)
        dist = 2.0 ** 0.5
        lattice_pt = np.zeros(2)
        grid = np.zeros(2)

        for j in range(-1, 2):
            grid[1] = j + n[1]

            for i in range(-1, 2):
                grid[0] = i + n[0]
                tiled_cell = self.modulo(grid, self.n_grid)
                jitter = self.hash22(tiled_cell)
                to_cell = grid + jitter - 0.5 - p

                if (length := self.get_norm(to_cell)) <= dist:
                    dist = length
                    lattice_pt[:] = tiled_cell + jitter

        return lattice_pt

    def vnoise3(self, p):
        """Args:
            p (numpy.ndarray) 3-element array
        """
        n = np.floor(p + 0.5)
        dist = 3.0 ** 0.5
        lattice_pt = np.zeros(3)
        grid = np.zeros(3)

        for k in range(-1, 2):
            grid[2] = k + n[2]

            for j in range(-1, 2):
                grid[1] = j + n[1]

                for i in range(-1, 2):
                    grid[0] = i + n[0]
                    tiled_cell = self.modulo(grid, self.n_grid)
                    jitter = self.hash33(tiled_cell)
                    to_cell = grid + jitter - 0.5 - p

                    if (length := self.get_norm(to_cell)) < dist:
                        dist = length
                        lattice_pt[:] = tiled_cell + jitter

        return lattice_pt

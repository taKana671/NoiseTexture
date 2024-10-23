import numpy as np

from noise import Noise


class Voronoi(Noise):

    def __init__(self, grid=4, size=256):
        self.size = size
        self.grid = grid

    def voronoi2(self, p):
        n = np.floor(p + 0.5)
        dist = 2.0 ** 0.5
        lattice_pt = np.zeros(2)

        for j in (0, 1, -1):
            if abs((y := n[1] + j) - p[1]) - 0.5 > dist:
                continue

            for i in range(-1, 2):
                x = n[0] + i
                grid = np.array([x, y])
                jitter = self.hash22(grid) - 0.5

                if (length := self.get_norm(grid + jitter - p)) <= dist:
                    dist = length
                    lattice_pt = grid

        return self.hash22(lattice_pt)

    def voronoi3(self, p):
        n = np.floor(p + 0.5)
        dist = 3.0 ** 0.5
        lattice_pt = np.zeros(0)

        for k in (0, 1, -1):
            if abs((z := n[2] + k) - p[2]) - 0.5 > dist:
                continue

            for j in (0, 1, -1):
                if abs((y := n[1] + j) - p[1]) - 0.5 > dist:
                    continue

                for i in range(-1, 2):
                    x = n[0] + i
                    grid = np.array([x, y, z])
                    jitter = self.hash33(grid) - 0.5

                    if (length := self.get_norm(grid + jitter - p)) <= dist:
                        dist = length
                        lattice_pt = grid

        return self.hash33(lattice_pt)

    def noise3(self, gray=True, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        if gray:
            vec = np.array([0.3, 0.6, 0.2])
            arr = np.array(
                [np.dot(self.voronoi3(np.array([x + t, y + t, t])), vec)
                    for y in np.linspace(0, self.grid, self.size)
                    for x in np.linspace(0, self.grid, self.size)]
            )
            arr = arr.reshape(self.size, self.size)
        else:
            arr = np.array(
                [self.voronoi3(np.array([x + t, y + t, t]))
                    for y in np.linspace(0, self.grid, self.size)
                    for x in np.linspace(0, self.grid, self.size)]
            )
            arr = arr.reshape(self.size, self.size, 3)

        return arr

    def noise2(self, gray=True, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        if gray:
            vec = np.array([0.3, 0.6])
            arr = np.array(
                [np.dot(self.voronoi2(np.array([x + t, y + t])), vec)
                    for y in np.linspace(0, self.grid, self.size)
                    for x in np.linspace(0, self.grid, self.size)]
            )
            arr = arr.reshape(self.size, self.size)
        else:
            arr = np.array(
                [[*self.voronoi2(np.array([x + t, y + t])), 1]
                    for y in np.linspace(0, self.grid, self.size)
                    for x in np.linspace(0, self.grid, self.size)]
            )
            arr = arr.reshape(self.size, self.size, 3)

        return arr
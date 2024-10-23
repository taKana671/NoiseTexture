import numpy as np

from noise import Noise


class Cellular(Noise):

    def __init__(self, grid=4, size=256):
        self.size = size
        self.grid = grid

    def sort4(self, dist4, length):

        if self.step(length, dist4[0]):
            return np.array([length, *dist4[:3]])

        if self.step(length, dist4[1]):
            return np.array([dist4[0], length, *dist4[1:3]])

        if self.step(length, dist4[2]):
            return np.array([*dist4[:2], length, dist4[3]])

        if self.step(length, dist4[3]):
            return np.array([*dist4[:3], length])

    def fdist24(self, p):
        """Compute the 1st, 2nd, 3rd and 4th 2D nearest neighbor distance.
            Args:
                p (numpy.numpy.ndarray): the length is 2.
        """
        n = np.floor(p + 0.5)
        temp = 1.5 - np.abs(p - n)
        length = sum(v ** 2 for v in temp) ** 0.5
        dist4 = np.full(4, length)

        for j in (0, 1, -1, 2, -2):
            if abs((y := n[1] + j) - p[1]) - 0.5 > dist4[3]:
                continue

            for i in range(-2, 3):
                x = n[0] + i
                grid = np.array([x, y])
                jitter = self.hash22(grid) - 0.5
                length = self.get_norm(grid + jitter - p)

                if (sorted_dist4 := self.sort4(dist4, length)) is not None:
                    dist4 = sorted_dist4

        return dist4

    def fdist2(self, p):
        """Compute 2D nearest neighbor distance.
            Args:
                p (numpy.numpy.ndarray): the length is 2.
        """
        n = np.floor(p + 0.5)
        dist = 2.0 ** 0.5

        for j in (0, 1, -1):
            if abs((y := n[1] + j) - p[1]) - 0.5 > dist:
                continue

            for i in range(-1, 2):
                x = n[0] + i
                grid = np.array([x, y])
                jitter = self.hash22(grid) - 0.5
                length = self.get_norm(grid + jitter - p)
                dist = min(dist, length)

        return dist

    def fdist3(self, p):
        """Compute 3D nearest neighbor distance.
            Args:
                p (numpy.numpy.ndarray): the length is 3.
        """
        n = np.floor(p + 0.5)
        dist = 3.0 ** 0.5

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
                    length = self.get_norm(grid + jitter - p)
                    dist = min(dist, length)

        return dist

    def noise2(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.fdist2(np.array([x + t, y + t]))
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def noise3(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.fdist3(np.array([x + t, y + t, t]))
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def noise24(self, nearest=2, t=None):
        """Return numpy.ndarray to be convert an 2D image.
            Args:
                nearest (int):
                    the order of nearest neighbor distance;
                    must be from 1 to 4.
        """
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.fdist24(np.array([x + t, y + t]))[nearest - 1]
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr
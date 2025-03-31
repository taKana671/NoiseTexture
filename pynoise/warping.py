import numpy as np


class Warping:

    def __init__(self, weight, octaves):
        self.weight = weight
        self.octaves = octaves


class DomainWarping2D(Warping):

    def __init__(self, noise_gen, weight=1.0, octaves=4):
        super().__init__(weight, octaves)
        self.noise = noise_gen

    def warp(self, x, y):
        """Args:
            x, y (float)
        """
        v = 0.0

        for _ in range(self.octaves):
            v = self.noise(x + self.weight * v, y + self.weight * v)

        return v

    def warp_rot(self, x, y):
        """Args:
            x, y (float)
        """
        v = 0.0

        for _ in range(self.octaves):
            xx = np.cos(2 * np.pi * v)
            yy = np.sin(2 * np.pi * v)
            v = self.noise(x + self.weight * xx, y + self.weight * yy)

        return v


class DomainWarping3D(Warping):

    def __init__(self, noise_gen, weight=1.0, octaves=4):
        super().__init__(weight, octaves)
        self.noise = noise_gen

    def warp(self, x, y, z):
        """Args:
            x, y, z (float)
        """
        v = 0.0

        for _ in range(self.octaves):
            v = self.noise(
                x + self.weight * v,
                y + self.weight * v,
                z + self.weight * v,
            )

        return v
import numpy as np


class DomainWarping:

    def __init__(self, noise_func, weight=1, octaves=4):
        self.noise = noise_func
        self.weight = 1
        self.octaves = octaves

    def warp(self, p):
        """Args:
            p (numpy.ndarray)
        """
        v = 0.0

        for i in range(self.octaves):
            v = self.noise(p + self.weight * v)

        return v

    def warp2_rot(self, p):
        """Args:
            p (numpy.ndarray)
        """
        v = 0.0

        for _ in range(self.octaves):
            arr = np.array([np.cos(2 * np.pi * v), np.sin(2 * np.pi * v)])
            v = self.noise(p + self.weight * arr)

        return v

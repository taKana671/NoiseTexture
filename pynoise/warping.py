import numpy as np


class DomainWarping:

    def __init__(self, noise_func, weight=1, octaves=4):
        self.noise = noise_func
        self.weight = weight
        self.octaves = octaves

    def warp(self, p):
        """Args:
            p (numpy.ndarray)
        """
        v = 0.0

        for _ in range(self.octaves):
            v = self.noise(p + self.weight * v)

        return v

    def warp_rot(self, p):
        """Args:
            p (numpy.ndarray)
        """
        v = 0.0

        for _ in range(self.octaves):
            arr = np.array([np.cos(2 * np.pi * v), np.sin(2 * np.pi * v)])
            v = self.noise(p + self.weight * arr)
            # ↑　いちいちnp.arrayを作らず、pの各要素に計算結果を足すほうがいいかも。

        return v

from abc import ABC, abstractmethod

import numpy as np

from .perlin import PerlinNoise
from .simplex import SimplexNoise


class CurlNoise3D(ABC):
    """A class to generate curl noise.
        Args:
            offset_1, offset2 (list):
              Offset the values significantly to avoid discontinuities.
              For example: offset_1 = [100, 200, 300], offset_2 = [500, 600, 700]
    """

    def __init__(self, offset_1, offset_2):
        self.off_1 = offset_1
        self.off_2 = offset_2

    @abstractmethod
    def vector_field_3d(self, x, y, z):
        """Generate a vector field.
            Args:
                x (float): x-coordinate of the vertex
                y (float): y-coordinate of the vertex
                z (float): z-coordinate of the vertex
        """
        pass

    def curl_3d(self, x, y, z, eps=0.0001):
        fx1 = self.vector_field_3d(x + eps, y, z)
        fx2 = self.vector_field_3d(x - eps, y, z)

        fy1 = self.vector_field_3d(x, y + eps, z)
        fy2 = self.vector_field_3d(x, y - eps, z)

        fz1 = self.vector_field_3d(x, y, z + eps)
        fz2 = self.vector_field_3d(x, y, z - eps)

        return np.array([
            (fy1[2] - fy2[2] - (fz1[1] - fz2[1])) / (2 * eps),
            (fz1[0] - fz2[0] - (fx1[2] - fx2[2])) / (2 * eps),
            (fx1[1] - fx2[1] - (fy1[0] - fy2[0])) / (2 * eps)
        ])


class PerlinCurlNoise3D(PerlinNoise, CurlNoise3D):
    """A class to genera 3D perlin curl noise."""

    def __init__(self, offset_1, offset_2):
        super().__init__(offset_1, offset_2)

    def vector_field_3d(self, x, y, z):
        return np.array([
            self.pnoise3(y, z, x),
            self.pnoise3(z + self.off_1[0], x + self.off_1[1], y + self.off_1[2]),
            self.pnoise3(x + self.off_2[0], y + self.off_2[1], z + self.off_2[2])
        ])


class SimplexCurlNoise3D(SimplexNoise, CurlNoise3D):
    """A class to genera 3D simplex curl noise."""

    def __init__(self, offset_1, offset_2):
        super().__init__(offset_1, offset_2)

    def vector_field_3d(self, x, y, z):
        return np.array([
            self.snoise3(y, z, x),
            self.snoise3(z + self.off_1[0], x + self.off_1[1], y + self.off_1[2]),
            self.snoise3(x + self.off_2[0], y + self.off_2[1], z + self.off_2[2])
        ])


class CurlNoise2D(ABC):

    @abstractmethod
    def vector_field_2d(self, x, y):
        """Generate a vector field.
            Args:
                x (float): x-coordinate of the vertex
                y (float): y-coordinate of the vertex
        """
        pass

    def curl_2d(self, x, y, eps=0.001):
        n1 = self.vector_field_2d(x + eps, y)
        n2 = self.vector_field_2d(x - eps, y)

        a = (n1 - n2) / (2 * eps)

        n1 = self.vector_field_2d(x, y + eps)
        n2 = self.vector_field_2d(x, y - eps)

        b = (n1 - n2) / (2 * eps)

        return np.array([b, -a])


class PerlinCurlNoise2D(PerlinNoise, CurlNoise2D):
    """A class to genera 3D perlin curl noise."""

    def vector_field_2d(self, x, y):
        return self.pnoise2(x, y)
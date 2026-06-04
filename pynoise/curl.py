import numpy as np


class CurlNoise:
    """A class to generate curl noise.
    """

    def __init__(self, noise_gen):
        self.noise = noise_gen


class CurlNoise3D(CurlNoise):
    """A class to genera 3D curl noise.
        Args:
            noise_gen (function): a function or method to generate noise.
            offset_1, offset2 (numpy.ndarray):
              Offset the values significantly to avoid discontinuities.
              For example: offset_1 = numpy.array([100, 200, 300]), offset_2 = numpy.array([500, 600, 700])
    """

    def __init__(self, noise_gen, offset_1, offset_2):
        super().__init__(noise_gen)
        self.off_1 = offset_1
        self.off_2 = offset_2

    def vector_field_3d(self, x, y, z):
        """Generate a vector field.
        Args:
            x (float): x-coordinate of the vertex
            y (float): y-coordinate of the vertex
            z (float): z-coordinate of the vertex
        """
        return np.array([
            self.noise(y, z, x),
            self.noise(z + self.off_1[0], x + self.off_1[1], y + self.off_1[2]),
            self.noise(x + self.off_2[0], y + self.off_2[1], z + self.off_2[2])
        ])

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


class CurlNoise2D(CurlNoise):

    def curl_2d(self, x, y, eps=0.001):
        n1 = self.noise(x + eps, y)
        n2 = self.noise(x - eps, y)

        a = (n1 - n2) / (2 * eps)

        n1 = self.noise(x, y + eps)
        n2 = self.noise(x, y - eps)

        b = (n1 - n2) / (2 * eps)

        return np.array([b, -a])

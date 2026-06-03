import numpy as np



class CurlNoise:

    def __init__(self, noise_gen):
        self.noise = noise_gen

    def create_vector_field(self, x, y, z, off_0=100, off_1=200, off_2=300,
                            off_3=400, off_4=500, off_5=600):
        return np.array([
            self.noise(y, z, x),
            self.noise(z + off_0, x + off_1, y + off_2),
            self.noise(x + off_3, y + off_4, z + off_5)
        ])

    def curl_3d(self, x, y, z, eps=0.0001):
        fx1 = self.create_vector_field(x + eps, y, z)
        fx2 = self.create_vector_field(x - eps, y, z)

        fy1 = self.create_vector_field(x, y + eps, z)
        fy2 = self.create_vector_field(x, y - eps, z)

        fz1 = self.create_vector_field(x, y, z + eps)
        fz2 = self.create_vector_field(x, y, z - eps)

        return np.array([
            (fy1[2] - fy2[2] - (fz1[1] - fz2[1])) / (2 * eps),
            (fz1[0] - fz2[0] - (fx1[2] - fx2[2])) / (2 * eps),
            (fx1[1] - fx2[1] - (fy1[0] - fy2[0])) / (2 * eps)
        ])

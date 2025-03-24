from libc.math cimport cos, sin, pi


cdef class Warping:

    def __init__(self, weight, octaves):
        self.weight = weight
        self.octaves = octaves


cdef class DomainWarping2D(Warping):

    def __init__(self, noise_gen, weight=1, octaves=4):
        super().__init__(weight, octaves)
        self.noise = noise_gen

    cdef double warp2(self, double x, double y):
        cdef:
            double v = 0.0

        for _ in range(self.octaves):
            v = self.noise(x + self.weight * v, y + self.weight * v)

        return v

    cdef double warp2_rot(self, double x, double y):
        cdef:
            double v = 0.0
            double xx, yy

        for _ in range(self.octaves):
            xx = cos(2.0 * pi * v)
            yy = sin(2.0 * pi * v)
            v = self.noise(x + self.weight * xx, y + self.weight * yy)

            # arr = np.array([np.cos(2 * np.pi * v), np.sin(2 * np.pi * v)])
            # v = self.noise(p + self.weight * arr)

        return v

    cpdef double warp(self, double x, double y):
        return self._warp2(x, y)

    cpdef double warp_rot(self, double x, double y):
        return self.warp2_rot(x, y)


cdef class DomainWarping3D(Warping):

    def __init__(self, noise_gen, weight=1, octaves=4):
        super().__init__(weight, octaves)
        self.noise = noise_gen

    cdef double warp3(self, double x, double y, double z):
        cdef:
            double v = 0.0

        for _ in range(self.octaves):
            v = self.noise(
                x + self.weight * v,
                y + self.weight * v,
                z + self.weight * v,
            )

        return v

    cpdef double warp(self, double x, double y, double z):
        return self.warp3(x, y, z)
# cython: language_level=3


cdef class Fractal:

    def __init__(self, gain, lacunarity, octaves):
        self.gain = gain
        self.lacunarity = lacunarity
        self.octaves = octaves


cdef class Fractal2D(Fractal):

    def __init__(self, noise_gen, gain=0.5, lacunarity=2.01, octaves=4):
        super().__init__(gain, lacunarity, octaves)
        self.noise = noise_gen

    cdef double _fractal2(self, double x, double y):
        cdef:
            double ret
            double v = 0.0
            double amp = 1.0
            double freq = 1.0

        for _ in range(self.octaves):
            ret = self.noise(freq * x, freq * y)
            v += amp * (ret - 0.5)
            amp *= self.gain
            freq *= self.lacunarity

        return 0.5 * v + 0.5

    
    cpdef double fractal(self, double x, double y):
        return self._fractal2(x, y)


cdef class Fractal3D(Fractal):

    def __init__(self, noise_gen, gain=0.5, lacunarity=2.01, octaves=4):
        super().__init__(gain, lacunarity, octaves)
        self.noise = noise_gen

    cdef double _fractal3(self, double x, double y, double z):
        cdef:
            double ret
            double v = 0.0
            double amp = 1.0
            double freq = 1.0

        for _ in range(self.octaves):
            ret = self.noise(freq * x, freq * y, freq * z)
            v += amp * (ret - 0.5)
            amp *= self.gain
            freq *= self.lacunarity

        return 0.5 * v + 0.5

    
    cpdef double fractal(self, double x, double y, double z):
        return self._fractal3(x, y, z)
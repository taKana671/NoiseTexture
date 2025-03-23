# cython: language_level=3


ctypedef double (*noise_gen2) (double, double)
ctypedef double (*noise_gen3) (double, double, double)


cdef class Fractal:

    cdef:
        # noise_gen2 noise_func 
        double gain
        double lacunarity
        int octaves
        

    # def __init__(self, noise_func, gain=0.5, lacunarity=2.01, octaves=4):
    def __init__(self, gain, lacunarity, octaves):
        # self.noise = noise_func
        self.gain = gain
        self.lacunarity = lacunarity
        self.octaves = octaves

    # def fractal(self, p):
    #     """Args:
    #         p (numpy.ndarray)
    #     """
    #     v = 0.0
    #     amp = 1.0          # amplitude: the highest deviation of the wave from its central or zero position
    #     freq = 1.0         # frequency: the number of waves that pass a fixed point in unit time

    #     for _ in range(self.octaves):
    #         v += amp * (self.noise(freq * p) - 0.5)
    #         amp *= self.gain
    #         freq *= self.lacunarity

    #     return 0.5 * v + 0.5



cdef class Fractal2:

    cdef:
        noise_gen2 noise_func 

    def __init__(self, noise_func, gain=0.5, lacunarity=2.01, octaves=4):
        super().__init__(gain, lacunarity, octaves)
        self.noise = noise_func

    cdef double fractal2(self, double x, double y):
        cdef:
            double v = 0.0
            double amp = 1.0
            double freq = 1.0

        for _ in range(self.octaves):
            v += amp * (self.noise(freq * x, freq * y) - 0.5)
            amp *= self.gain
            freq *= self.lacunarity

        return 0.5 * v + 0.5

    
    cpdef double fractal(self, double x, double y):
        return self.fractal2(x, y)


cdef class Fractal3:

    cdef:
        noise_gen3 noise_func 

    def __init__(self, noise_func, gain=0.5, lacunarity=2.01, octaves=4):
        super().__init__(gain, lacunarity, octaves)
        self.noise = noise_func

    cdef double fractal3(self, double x, double y, double z):
        cdef:
            double v = 0.0
            double amp = 1.0
            double freq = 1.0

        for _ in range(self.octaves):
            v += amp * (self.noise(freq * x, freq * y, freq * z) - 0.5)
            amp *= self.gain
            freq *= self.lacunarity

        return 0.5 * v + 0.5

    
    cpdef double fractal(self, double x, double y, double z):
        return self.fractal3(x, y, z)
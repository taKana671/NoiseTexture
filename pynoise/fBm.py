

class Fractal:

    def __init__(self, noise_func, gain=0.5, lacunarity=2.01, octaves=4):
        self.noise = noise_func
        self.gain = gain
        self.lacunarity = lacunarity
        self.octaves = octaves

    def fractal(self, p):
        """Args:
            p (numpy.ndarray)
        """
        v = 0.0
        amp = 1.0          # amplitude: the highest deviation of the wave from its central or zero position
        freq = 1.0         # frequency: the number of waves that pass a fixed point in unit time

        for _ in range(self.octaves):
            v += amp * (self.noise(freq * p) - 0.5)
            amp *= self.gain
            freq *= self.lacunarity

        return 0.5 * v + 0.5

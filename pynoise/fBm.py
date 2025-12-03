

class Fractal:

    def __init__(self, gain, lacunarity, octaves, amplitude, frequency):
        self.gain = gain
        self.lacunarity = lacunarity
        self.octaves = octaves
        self.amplitude = amplitude
        self.frequency = frequency


class Fractal2D(Fractal):

    def __init__(self, noise_gen, gain=0.5, lacunarity=2.01, octaves=4,
                 amplitude=1.0, frequency=1.0):
        super().__init__(gain, lacunarity, octaves, amplitude, frequency)
        self.noise = noise_gen

    def fractal(self, x, y):
        v = 0.0
        amp = self.amplitude     # amplitude: the highest deviation of the wave from its central or zero position
        freq = self.frequency    # frequency: the number of waves that pass a fixed point in unit time

        for _ in range(self.octaves):
            v += amp * (self.noise(freq * x, freq * y) - 0.5)
            amp *= self.gain
            freq *= self.lacunarity

        return 0.5 * v + 0.5

    def noise_octaves(self, x, y):
        """Apply multiple passes of noise algorithm, each one with a higher frequency
           than the previous one, but with a lower “strength”.
        """
        v = 0.0
        amp = self.amplitude
        freq = self.frequency

        for _ in range(self.octaves):
            noise = self.noise(freq * x, freq * y)
            v += amp * noise
            freq *= self.lacunarity
            amp *= self.gain

        return v


class Fractal3D(Fractal):

    def __init__(self, noise_gen, gain=0.5, lacunarity=2.01, octaves=4,
                 amplitude=1.0, frequency=1.0):
        super().__init__(gain, lacunarity, octaves, amplitude, frequency)
        self.noise = noise_gen

    def fractal(self, x, y, z):
        v = 0.0
        amp = self.amplitude    # amplitude: the highest deviation of the wave from its central or zero position
        freq = self.frequency   # frequency: the number of waves that pass a fixed point in unit time

        for _ in range(self.octaves):
            v += amp * (self.noise(freq * x, freq * y, freq * z) - 0.5)
            amp *= self.gain
            freq *= self.lacunarity

        return 0.5 * v + 0.5

    def noise_octaves(self, x, y, z):
        """Apply multiple passes of noise algorithm, each one with a higher frequency
           than the previous one, but with a lower “strength”.
        """
        v = 0.0
        amp = self.amplitude
        freq = self.frequency

        for _ in range(self.octaves):
            noise = self.noise(freq * x, freq * y, freq * z)
            v += amp * noise
            freq *= self.lacunarity
            amp *= self.gain

        return v
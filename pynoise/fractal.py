from pynoise.noise import Noise


class Fractal(Noise):

    def __init__(self, noise_func, weight=0.5, lacunarity=2.01, octaves=4):
        self.noise = noise_func
        self.weight = weight
        self.lacunarity = lacunarity
        self.octaves = octaves

    def fractal(self, p):
        v = 0.0
        amp = 1.0
        freq = 1.0

        for _ in range(self.octaves):
            v += amp * (self.noise(freq * p) - 0.5)
            amp *= self.weight
            freq *= self.lacunarity

        return 0.5 * v + 0.5

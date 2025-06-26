

class Fractal:

    def __init__(self, gain, lacunarity, octaves):
        self.gain = gain
        self.lacunarity = lacunarity
        self.octaves = octaves


class Fractal2D(Fractal):

    def __init__(self, noise_gen, gain=0.5, lacunarity=2.01, octaves=4):
        super().__init__(gain, lacunarity, octaves)
        self.noise = noise_gen

    def fractal(self, x, y):
        v = 0.0
        amp = 1.0          # amplitude: the highest deviation of the wave from its central or zero position
        freq = 1.0         # frequency: the number of waves that pass a fixed point in unit time

        for _ in range(self.octaves):
            v += amp * (self.noise(freq * x, freq * y) - 0.5)
            amp *= self.gain
            freq *= self.lacunarity

        return 0.5 * v + 0.5


class Fractal3D(Fractal):

    def __init__(self, noise_gen, gain=0.5, lacunarity=2.01, octaves=4):
        super().__init__(gain, lacunarity, octaves)
        self.noise = noise_gen

    def fractal(self, x, y, z):
        v = 0.0
        amp = 1.0          # amplitude: the highest deviation of the wave from its central or zero position
        freq = 1.0         # frequency: the number of waves that pass a fixed point in unit time

        for _ in range(self.octaves):
            v += amp * (self.noise(freq * x, freq * y, freq * z) - 0.5)
            amp *= self.gain
            freq *= self.lacunarity

        return 0.5 * v + 0.5


# class Fractal4D(Fractal):

#     def __init__(self, noise_gen, gain=0.5, lacunarity=2.01, octaves=4):
#         super().__init__(gain, lacunarity, octaves)
#         self.noise = noise_gen

#     def fractal(self, x, y, z, w):
#         v = 0.0
#         amp = 1.0          # amplitude: the highest deviation of the wave from its central or zero position
#         freq = 1.0         # frequency: the number of waves that pass a fixed point in unit time

#         for _ in range(self.octaves):
#             v += amp * (self.noise(freq * x, freq * y, freq * z, freq * w) - 0.5)
#             amp *= self.gain
#             freq *= self.lacunarity

#         return 0.5 * v + 0.5

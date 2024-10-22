import cv2
import numpy as np

from noise import Noise


class FractionalBrownianMotion(Noise):

    def __init__(self, weight=0.5, grid=4, size=256):
        self.size = size
        self.grid = grid
        self.weight = weight

    def vnoise2(self, p):
        n = np.floor(p)
        v = [self.hash21(n + np.array([i, j])) for j in range(2) for i in range(2)]

        f, _ = np.modf(p)
        f = 6 * f**5 - 15 * f**4 + 10 * f**3
        w0 = self.mix(v[0], v[1], f[0])
        w1 = self.mix(v[2], v[3], f[0])
        return self.mix(w0, w1, f[1])

    def fbm2(self, p):
        v = 0.0
        amp = 1.0
        freq = 1.0

        for i in range(4):
            v += amp * (self.vnoise2(freq * p) - 0.5)
            amp *= self.weight
            freq *= 2.011

        return 0.5 * v + 0.5

    def noise2(self, t=None):
        t = self.mock_time() if t is None else t
        self.hash = {}

        arr = np.array(
            [self.fbm2(np.array([x + t, y + t]))
                for y in np.linspace(0, self.grid, self.size)
                for x in np.linspace(0, self.grid, self.size)]
        )
        arr = arr.reshape(self.size, self.size)
        return arr

    def wrap2(self, x, y, rot=False):
        v = 0.0

        for i in range(4):
            cx = np.cos(2 * np.pi * v) if rot else v
            sy = np.sin(2 * np.pi * v) if rot else v
            x += self.weight * cx
            y += self.weight * sy
            v = self.fbm2(np.array([x, y]))

        return v


# np.count_nonzero(np.sign(arr) < 0) ; no less than zero: no
def create_img_8bit(path, weight=0.5, grid=4, size=256):
    fbm = FractionalBrownianMotion(weight, grid, size)
    # fbm.g = 4.0
    arr = fbm.noise2()
    # arr = fbm.wrap(rot=False)
    # arr = fbm.convert_gradation(rot=True)

    arr *= 255
    arr = arr.astype(np.uint8)
    cv2.imwrite(path, arr)


def create_img_16bit(path, grid=4, size=256):
    fbm = FractionalBrownianMotion(grid, size)
    arr = fbm.noise2()

    arr *= 65535
    arr = arr.astype(np.uint16)
    cv2.imwrite(path, arr)


if __name__ == '__main__':
    # create_img_8bit('fbm_sample02.png')
    create_img_8bit('test2.png')
    # create_img_16bit('fbm_sample02.png')
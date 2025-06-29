import random
from functools import wraps

import numpy as np


k = np.array([1164413355, 1737075525, 2309703015, 2873452425], dtype=np.uint32)
u = np.array([1, 2, 3, 4], dtype=np.uint32)
UINT_MAX = np.iinfo(np.uint32).max


def cache(max_length):
    def decoration(func):
        dic = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(dic) > max_length:
                dic.clear()

            if (key := tuple(args[1])) in dic:
                return dic[key]

            ret = func(*args, **kwargs)
            dic[key] = ret
            return ret

        return wrapper
    return decoration


class Noise:

    def mock_time(self):
        return random.uniform(0, 1000)

    def get_4_nums(self, is_rnd=True):
        if is_rnd:
            li = random.sample(list('123456789'), 4)
            sub = li[:3]

            aa = int(''.join(sub))
            bb = int(''.join([sub[1], sub[2], sub[0]]))
            cc = int(''.join(sub[::-1]))
            dd = int(''.join([sub[1], li[3], sub[2]]))

            return aa, bb, cc, dd

        return 123, 231, 321, 273

    def uhash11(self, n):
        n ^= n << u[0]
        n ^= n >> u[0]
        n *= k[0]
        n ^= n << u[0]
        n *= k[0]

    def uhash22(self, n):
        n ^= n[::-1] << u[:2]
        n ^= n[::-1] >> u[:2]
        n *= k[:2]
        n ^= n[::-1] << u[:2]
        n *= k[:2]

    def uhash33(self, n):
        n ^= n[[1, 2, 0]] << u[:3]
        n ^= n[[1, 2, 0]] >> u[:3]
        n *= k[:3]
        n ^= n[[1, 2, 0]] << u[:3]
        n *= k[:3]

    def uhash44(self, n):
        n ^= n[[1, 2, 3, 0]] << u
        n ^= n[[1, 2, 3, 0]] >> u
        n *= k
        n ^= n[[1, 2, 3, 0]] << u
        n *= k

    @cache(128)
    def hash21(self, p):
        n = p.astype(np.uint32)
        self.uhash22(n)
        h = n[0] / UINT_MAX

        return h

    @cache(128)
    def hash31(self, p):
        n = p.astype(np.uint32)
        self.uhash33(n)
        h = n[0] / UINT_MAX

        return h

    @cache(128)
    def hash22(self, p):
        n = p.astype(np.uint32)
        self.uhash22(n)
        h = n / UINT_MAX

        return h

    @cache(128)
    def hash33(self, p):
        n = p.astype(np.uint32)
        self.uhash33(n)
        h = n / UINT_MAX

        return h

    @cache(128)
    def hash44(self, p):
        n = p.astype(np.uint32)
        self.uhash44(n)
        h = n / UINT_MAX

        return h

    def mix(self, x, y, a):
        return x + (y - x) * a

    def step(self, edge, x):
        if x < edge:
            return 0
        return 1

    def smoothstep(self, edge0, edge1, x):
        """Args:
            edge0, edge1, x (float)
        """
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def xy2pol(self, x, y):
        r = (x ** 2 + y ** 2) ** 0.5

        if x == 0:
            x = np.sign(y) * np.pi / 2
        else:
            x = np.arctan2(y, x)

        return x, r

    def get_norm(self, vec):
        return sum(v ** 2 for v in vec) ** 0.5

    def hermite_interpolation(self, p):
        return 3 * p**2 - 2 * p**3

    def quintic_hermite_interpolation(self, x):
        return 6 * x**5 - 15 * x**4 + 10 * x**3

    def clamp(self, x, min_val, max_val):
        """Args:
            x (float): the value to constrain.
            min_val (float): the lower end of the range into which to constrain x.
            max_val (float): the upper end of the range into which to constrain x.
        """
        return min(max(x, min_val), max_val)

    def normalize(self, p):
        if (norm := np.sqrt(np.sum(p**2))) == 0:
            norm = 1

        return p / norm
import random

import cython
# cimport numpy as np
# import numpy as np
from libc.math cimport atan2, cos, sin, pi, floor, fmax, fmin


cdef class Noise:

    def __init__(self):
        # self.k = [1164413355, 1737075525, 2309703015, 2873452425]
        self.k = [0x456789abu, 0x6789ab45u, 0x89ab4567u, 0xab456789u]
        self.u = [1, 2, 3, 4]
        self.uint_max = 4294967295

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

    cdef unsigned int uhash11(self, unsigned int n):
        # I did not know why, using array pointers made all return values 0.
        n ^= (n << 1)
        n ^= (n >> 1)
        n *= 0x456789abu
        n ^= (n << 1)
        n *= 0x456789abu
        return n

    cdef void uhash22(self, unsigned int[2] *n):
        cdef:
            unsigned int[2] tmp

        tmp = n[0]
        n[0][0] ^= tmp[1] << self.u[0]
        n[0][1] ^= tmp[0] << self.u[1]

        tmp = n[0]
        n[0][0] ^= tmp[1] >> self.u[0]
        n[0][1] ^= tmp[0] >> self.u[1]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]

        tmp = n[0]
        n[0][0] ^= tmp[1] << self.u[0]
        n[0][1] ^= tmp[0] << self.u[1]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]

    cdef void uhash33(self, unsigned int[3] *n):
        cdef:
            unsigned int[3] tmp

        tmp = n[0]
        n[0][0] ^= tmp[1] << self.u[0]
        n[0][1] ^= tmp[2] << self.u[1]
        n[0][2] ^= tmp[0] << self.u[2]

        tmp = n[0]
        n[0][0] ^= tmp[1] >> self.u[0]
        n[0][1] ^= tmp[2] >> self.u[1]
        n[0][2] ^= tmp[0] >> self.u[2]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]
        n[0][2] *= self.k[2]

        tmp = n[0]
        n[0][0] ^= tmp[1] << self.u[0]
        n[0][1] ^= tmp[2] << self.u[1]
        n[0][2] ^= tmp[0] << self.u[2] 

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]
        n[0][2] *= self.k[2]

    cdef void uhash44(self, unsigned int[4] *n):
        cdef:
            unsigned int[4] tmp

        tmp = n[0]
        n[0][0] ^= tmp[1] << self.u[0]
        n[0][1] ^= tmp[2] << self.u[1]
        n[0][2] ^= tmp[3] << self.u[2]
        n[0][3] ^= tmp[0] << self.u[3]

        tmp = n[0]
        n[0][0] ^= tmp[1] >> self.u[0]
        n[0][1] ^= tmp[2] >> self.u[1]
        n[0][2] ^= tmp[3] >> self.u[2]
        n[0][3] ^= tmp[0] >> self.u[3]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]
        n[0][2] *= self.k[2]
        n[0][3] *= self.k[3]

        tmp = n[0]
        n[0][0] ^= tmp[1] << self.u[0]
        n[0][1] ^= tmp[2] << self.u[1]
        n[0][2] ^= tmp[3] << self.u[2] 
        n[0][3] ^= tmp[0] << self.u[3]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]
        n[0][2] *= self.k[2]
        n[0][3] *= self.k[3]

    @cython.cdivision(True)
    cdef double hash21(self, double[2] *p):
        cdef:
            unsigned int i
            unsigned int[2] n
            double h

        for i in range(2):
            n[i] = <unsigned int>p[0][i]

        self.uhash22(&n)
        h = <double>n[0] / self.uint_max
        return h
    
    @cython.cdivision(True)
    cdef double hash31(self, double[3] *p):
        cdef:
            unsigned int i
            unsigned int[3] n
            double h
        
        for i in range(3):
            n[i] = <unsigned int>p[0][i]
        
        self.uhash33(&n)
        h = <double>n[0] / self.uint_max
        return h

    @cython.cdivision(True)
    cdef void hash22(self, double[2] *p, double[2] *h):
        cdef:
            unsigned int i
            unsigned int[2] n

        for i in range(2):
            n[i] = <unsigned int>p[0][i]

        self.uhash22(&n)

        for i in range(2):
            h[0][i] = <double>n[i] / self.uint_max

    @cython.cdivision(True)
    cdef void hash33(self, double[3] *p, double[3] *h):
        cdef:
            unsigned int i
            unsigned int[3] n

        for i in range(3):
            n[i] = <unsigned int>p[0][i]

        self.uhash33(&n)

        for i in range(3):
            h[0][i] = <double>n[i] / self.uint_max

    @cython.cdivision(True)
    cdef void hash44(self, double[4] *p, double[4] *h):
        cdef:
            unsigned int i
            unsigned int[4] n

        for i in range(4):
            n[i] = <unsigned int>p[0][i]

        self.uhash44(&n)

        for i in range(4):
            h[0][i] = <double>n[i] / self.uint_max

    cdef double hermite_interpolation(self, double x):
        return 3 * x**2 - 2 * x**3

    cdef double quintic_hermite_interpolation(self, double x):
        return 6 * x**5 - 15 * x**4 + 10 * x**3

    cdef double mix(self, double x, double y, double a):
        return x + (y - x) * a

    cdef void mix3(self, double[3] *x, double[3] *y, double a, double[3] *m):
        cdef:
            int i

        for i in range(3):
            m[0][i] = x[0][i] + (y[0][i] - x[0][i]) * a

    cdef unsigned int step(self, double a, double x):
        if x < a:
            return 0
        return 1

    cdef double clamp(self, double x, double a, double b):
        return fmin(fmax(x, a), b)

    @cython.cdivision(True)
    cdef double smoothstep(self, double edge0, double edge1, double x):
        t = self.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @cython.cdivision(True)
    cdef double sign_with_abs(self, double *x):
        if abs(x[0]) == 0:
            return 0.0
   
        return x[0] / abs(x[0])
    
    @cython.cdivision(True)
    cdef (double, double) xy2pol(self, double x, double y):
        cdef:
            double r, px

        r = (x ** 2 + y ** 2) ** 0.5

        if x == 0:
            px = self.sign_with_abs(&y) * pi / 2.0
        else:
            px = atan2(y, x)

        return px, r

    cdef double get_norm3(self, double[3] *v):
        return (v[0][0] ** 2 + v[0][1] ** 2 + v[0][2] ** 2) ** 0.5

    cdef double get_norm2(self, double[2] *v):
        return (v[0][0] ** 2 + v[0][1] ** 2) ** 0.5

    @cython.cdivision(True)
    cdef double mod(self, double x, double y):
        return x - y * floor(x / y)
    
    cdef double inner_product22(self, double[2] *arr1, double[2] *arr2):
        cdef:
            double inner = 0
            unsigned int i
        
        for i in range(2):
            inner += arr1[0][i] * arr2[0][i]

        return inner
    
    cdef double inner_product33(self, double[3] *arr1, double[3] *arr2):
        cdef:
            double inner = 0
            unsigned int i
        
        for i in range(3):
            inner += arr1[0][i] * arr2[0][i]

        return inner
    
    cdef double inner_product44(self, double[4] *arr1, double[4] *arr2):
        cdef:
            double inner = 0
            unsigned int i
        
        for i in range(4):
            inner += arr1[0][i] * arr2[0][i]

        return inner
    
    cdef double inner_product21(self, double[2] *arr, double *v):
        cdef:
            double inner = 0
            unsigned int i
        
        for i in range(2):
            inner += arr[0][i] * v[0]

        return inner
    
    cdef double inner_product31(self, double[3] *arr, double *v):
        cdef:
            double inner = 0
            unsigned int i
        
        for i in range(3):
            inner += arr[0][i] * v[0]

        return inner

    cdef double inner_product41(self, double[4] *arr, double *v):
        cdef:
            double inner = 0
            unsigned int i
        
        for i in range(4):
            inner += arr[0][i] * v[0]

        return inner
    
    @cython.cdivision(True)
    cdef void normalize2(self, double[2] *p, double[2] *nm):
        cdef:
            double norm
            int i
        
        norm = self.get_norm2(p)
        if norm == 0.0:
            norm = 1.0

        for i in range(2):
            nm[0][i] = p[0][i] / norm
    
    @cython.cdivision(True)
    cdef void normalize3(self, double[3] *p, double[3] *nm):
        cdef:
            double norm
            int i
        
        norm = self.get_norm3(p)
        if norm == 0.0:
            norm = 1.0

        for i in range(3):
            nm[0][i] = p[0][i] / norm

    cdef void modulo21(self, double[2] *divident, double *divisor, double[2] *m):
        cdef:
            double d
            unsigned int i

        for i in range(2):
            d = self.mod(divident[0][i], divisor[0]) + divisor[0]
            m[0][i] = self.mod(d, divisor[0])

    cdef void modulo31(self, double[3] *divident, double *divisor, double[3] *m):
        cdef:
            double d
            unsigned int i

        for i in range(3):
            d = self.mod(divident[0][i], divisor[0]) + divisor[0]
            m[0][i] = self.mod(d, divisor[0])


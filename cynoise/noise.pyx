# cython: language_level=3

import random

import cython
# cimport numpy as np
import numpy as np
from libc.math cimport atan2, cos, sin, pi, floor


cdef class Noise:

    def __init__(self):
        # self.k = [1164413355, 1737075525, 2309703015]
        self.k = [0x456789abu, 0x6789ab45u, 0x89ab4567u]
        self.u = [1, 2, 3]
        self.uint_max = 4294967295

    def mock_time(self):
        return random.uniform(0, 1000)

    cdef unsigned int uhash11(self, unsigned int n):
        # I did not know why, using array pointers made all return values 0.
        n ^= (n << 1)
        n ^= (n >> 1)
        n *= 0x456789abu
        n ^= (n << 1)
        n *= 0x456789abu
        return n

    cdef void uhash22(self, unsigned int[2] *n):
        n[0][0] ^= n[0][1] << self.u[0]
        n[0][1] ^= n[0][0] << self.u[1]

        n[0][0] ^= n[0][1] >> self.u[0]
        n[0][1] ^= n[0][0] >> self.u[1]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]

        n[0][0] ^= n[0][1] << self.u[0]
        n[0][1] ^= n[0][0] << self.u[1]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]

    cdef void uhash33(self, unsigned int[3] *n):
        n[0][0] ^= n[0][1] << self.u[0]
        n[0][1] ^= n[0][2] << self.u[1]
        n[0][2] ^= n[0][0] << self.u[2]

        n[0][0] ^= n[0][1] >> self.u[0]
        n[0][1] ^= n[0][2] >> self.u[1]
        n[0][2] ^= n[0][0] >> self.u[2]

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]
        n[0][2] *= self.k[2]

        n[0][0] ^= n[0][1] << self.u[0]
        n[0][1] ^= n[0][2] << self.u[1]
        n[0][2] ^= n[0][0] << self.u[2] 

        n[0][0] *= self.k[0]
        n[0][1] *= self.k[1]
        n[0][2] *= self.k[2]

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

    cdef double hermite_interpolation(self, double x):
        return 3 * x**2 - 2 * x**3

    cdef double quintic_hermite_interpolation(self, double x):
        return 6 * x**5 - 15 * x**4 + 10 * x**3

    cdef double mix(self, double x, double y, double a):
        return x + (y - x) * a

    cdef unsigned int step(self, double a, double x):
        # if x <= a:
        if x < a:
            return 0
        return 1

    def smoothstep(self, edge0, edge1, x):
        """Args:
            edge0, edge1, x (float)
        """
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
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
    
    cdef double inner_product31(self, double[3] *arr, double *v):
        cdef:
            double inner = 0
            unsigned int i
        
        for i in range(3):
            inner += arr[0][i] * v[0]

        return inner

    cdef double inner_product21(self, double[2] *arr, double *v):
        cdef:
            double inner = 0
            unsigned int i
        
        for i in range(2):
            inner += arr[0][i] * v[0]

        return inner

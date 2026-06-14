from .noise cimport Noise


cdef class SimplexNoise(Noise):

    cdef double mod289(self, double v)

    cdef double permute(self, double v)

    cdef double inverssqrt(self, double v)

    cdef double _snoise2(self, double x, double y)

    cdef void product31(self, double[3] *arr, double v)

    cdef double _snoise3(self, double x, double y, double z)

    cdef double _snoise4(self, double x, double y, double z, double w)

    cdef void grad4(self, double *j, double[3] *ip, double[4] *p)

    cpdef double snoise2(self, double x, double y)

    cpdef double snoise3(self, double x, double y, double z)

    cpdef double snoise4(self, double x, double y, double z, double w)

    cpdef noise2(self, width=*, height=*, scale=*, t=*)

    cpdef noise3(self, width=*, height=*, scale=*, t=*)

    cpdef noise4(self, width=*, height=*, scale=*, t=*)

    cpdef fractal2(self, width=*, height=*, t=*, gain=*, lacunarity=*, octaves=*)

    cpdef fractal3(self, width=*, height=*, t=*, gain=*, lacunarity=*, octaves=*)
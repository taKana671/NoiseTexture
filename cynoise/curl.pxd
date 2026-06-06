from .perlin cimport PerlinNoise
from .simplex cimport SimplexNoise


cdef class PerlinCurlNoise3D(PerlinNoise):

    cdef:
        double[3] *off_1
        double[3] *off_2

    cdef void _vector_field_3d(self, double x, double y, double z, double[3] *arr)

    cdef list _curl_3d(self, double x, double y, double z, double eps)

    cpdef curl_3d(self, double x, double y, double z, double eps=*)


cdef class SimplexCurlNoise3D(SimplexNoise):

    cdef:
        double[3] *off_1
        double[3] *off_2

    cdef void _vector_field_3d(self, double x, double y, double z, double[3] *arr)

    cdef list _curl_3d(self, double x, double y, double z, double eps)

    cpdef curl_3d(self, double x, double y, double z, double eps=*)


cdef class PerlinCurlNoise2D(PerlinNoise):

    cdef list _curl_2d(self, double x, double y, double eps)

    cpdef curl_2d(self, double x, double y, double eps=*)



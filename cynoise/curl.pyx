import cython
import numpy as np

from .perlin cimport PerlinNoise
from .simplex cimport SimplexNoise


cdef class PerlinCurlNoise3D(PerlinNoise):
    """A class to genera 3D perlin curl noise.
        Args:
            offset_1, offset_2 (list)
              offset_1 and offset_2 are used to offset the values significantly to avoid discontinuities.
              For example: offset_1 = [100, 200, 300], offset_2 = [500, 600, 700]
    """

    def __init__(self, offset_1, offset_2):
        super().__init__()

    def __cinit__(self, list offset_1, list offset_2):
        cdef:
            double[3] offset_arr_1
            double[3] offset_arr_2
            int i

        if len(offset_1) != 3 or len(offset_2) != 3:
            raise MemoryError('The length of offset_* must be 3.')

        for i in range(2):
            offset_arr_1[i] = <double>offset_1[i]
            offset_arr_2[i] = <double>offset_2[i]
        
        self.off_1 = &offset_arr_1
        self.off_2 = &offset_arr_2

    cdef void _vector_field_3d(self, double x, double y, double z, double[3] *arr):
        arr[0][0] = self._pnoise3(y, z, x)
        arr[0][1] = self._pnoise3(z + self.off_1[0][0], x + self.off_1[0][1], y + self.off_1[0][2])
        arr[0][2] = self._pnoise3(x + self.off_2[0][0], y + self.off_2[0][1], z + self.off_2[0][2]) 

    @cython.cdivision(True)
    cdef list _curl_3d(self, double x, double y, double z, double eps):
        cdef:
            double[3] fx1, fx2, fy1, fy2, fz1, fz2
            double[3] arr

        self._vector_field_3d(x + eps, y, z, &fx1)
        self._vector_field_3d(x - eps, y, z, &fx2)

        self._vector_field_3d(x, y + eps, z, &fy1)
        self._vector_field_3d(x, y - eps, z, &fy2)

        self._vector_field_3d(x, y, z + eps, &fz1)
        self._vector_field_3d(x, y, z - eps, &fz2)

        arr[0] = (fy1[2] - fy2[2] - (fz1[1] - fz2[1])) / (2.0 * eps)
        arr[1] = (fz1[0] - fz2[0] - (fx1[2] - fx2[2])) / (2.0 * eps)
        arr[2] = (fx1[1] - fx2[1] - (fy1[0] - fy2[0])) / (2.0 * eps)

        return arr

    cpdef curl_3d(self, double x, double y, double z, double eps=0.0001):
        return np.array(self._curl_3d(x, y, z, eps))


cdef class SimplexCurlNoise3D(SimplexNoise):
    """A class to genera 3D perlin curl noise.
        Args:
            offset_1, offset_2 (list)
              offset_1 and offset_2 are used to offset the values significantly to avoid discontinuities.
              For example: offset_1 = [100, 200, 300], offset_2 = [500, 600, 700]
    """

    def __init__(self, offset_1, offset_2):
        super().__init__()

    def __cinit__(self, list offset_1, list offset_2):
        cdef:
            double[3] offset_arr_1
            double[3] offset_arr_2
            int i

        if len(offset_1) != 3 or len(offset_2) != 3:
            raise MemoryError('The length of offset_* must be 3.')

        for i in range(2):
            offset_arr_1[i] = <double>offset_1[i]
            offset_arr_2[i] = <double>offset_2[i]
        
        self.off_1 = &offset_arr_1
        self.off_2 = &offset_arr_2

    cdef void _vector_field_3d(self, double x, double y, double z, double[3] *arr):
        arr[0][0] = self._snoise3(y, z, x)
        arr[0][1] = self._snoise3(z + self.off_1[0][0], x + self.off_1[0][1], y + self.off_1[0][2])
        arr[0][2] = self._snoise3(x + self.off_2[0][0], y + self.off_2[0][1], z + self.off_2[0][2]) 

    @cython.cdivision(True)
    cdef list _curl_3d(self, double x, double y, double z, double eps):
        cdef:
            double[3] fx1, fx2, fy1, fy2, fz1, fz2
            double[3] arr

        self._vector_field_3d(x + eps, y, z, &fx1)
        self._vector_field_3d(x - eps, y, z, &fx2)

        self._vector_field_3d(x, y + eps, z, &fy1)
        self._vector_field_3d(x, y - eps, z, &fy2)

        self._vector_field_3d(x, y, z + eps, &fz1)
        self._vector_field_3d(x, y, z - eps, &fz2)

        arr[0] = (fy1[2] - fy2[2] - (fz1[1] - fz2[1])) / (2.0 * eps)
        arr[1] = (fz1[0] - fz2[0] - (fx1[2] - fx2[2])) / (2.0 * eps)
        arr[2] = (fx1[1] - fx2[1] - (fy1[0] - fy2[0])) / (2.0 * eps)

        return arr

    cpdef curl_3d(self, double x, double y, double z, double eps=0.0001):
        return np.array(self._curl_3d(x, y, z, eps))        



cdef class PerlinCurlNoise2D(PerlinNoise):

    @cython.cdivision(True)
    cdef list _curl_2d(self, double x, double y, double eps):
        cdef:
            double n1, n2
            double[2] arr

        n1 = self._pnoise2(x + eps, y)
        n2 = self._pnoise2(x - eps, y)
        arr[1] = -((n1 - n2) / (2.0 * eps)) 

        n1 = self._pnoise2(x, y + eps)
        n2 = self._pnoise2(x, y - eps)
        arr[0] = (n1 - n2) / (2.0 * eps)

        return arr

    cpdef curl_2d(self, double x, double y, double eps=0.001):
        return np.array(self._curl_2d(x, y, eps))

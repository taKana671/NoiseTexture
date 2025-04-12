# cython: language_level=3


cdef class Noise:

    cdef:
        unsigned int[3] k
        unsigned int[3] u
        unsigned int uint_max

    cdef unsigned int uhash11(self, unsigned int n)

    cdef void uhash22(self, unsigned int[2] *n)

    cdef void uhash33(self, unsigned int[3] *n)

    cdef double hash21(self, double[2] *p)

    cdef double hash31(self, double[3] *p)

    cdef void hash22(self, double[2] *p, double[2] *h)

    cdef void hash33(self, double[3] *p, double[3] *h)

    cdef double hermite_interpolation(self, double x)

    cdef double quintic_hermite_interpolation(self, double x)

    cdef double mix(self, double x, double y, double a)

    cdef double get_norm3(self, double[3] *v)

    cdef double get_norm2(self, double[2] *v)

    cdef unsigned int step(self, double a, double x)

    cdef double sign_with_abs(self, double *x)

    cdef (double, double) xy2pol(self, double x, double y)

    cdef double mod(self, double x, double y)

    cdef double inner_product22(self, double[2] *arr1, double[2] *arr2)

    cdef double inner_product33(self, double[3] *arr1, double[3] *arr2)

    cdef double inner_product44(self, double[4] *arr1, double[4] *arr2)

    cdef double inner_product31(self, double[3] *arr, double *v)

    cdef double inner_product21(self, double[2] *arr, double *v)

    cdef double clamp(self, double x, double a, double b)

    cdef double smoothstep(self, double edge0, double edge1, double x)

    cdef void normalize2(self, double[2] *p, double[2] *nm)

    cdef void normalize3(self, double[3] *p, double[3] *nm)

    cdef void mix3(self, double[3] *x, double[3] *y, double a, double[3] *m)
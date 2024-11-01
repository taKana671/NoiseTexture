# cython: language_level=3


cdef class Noise:

    cdef:
        int grid
        int size
        unsigned int[3] k
        unsigned int[3] u

    cdef void uhash11(self, unsigned int *n)

    cdef void uhash22(self, unsigned int[2] *n)

    cdef void uhash33(self, unsigned int[3] *n)

    cdef double hash21(self, double[2] *p)

    cdef void hash22(self, double[2] *p, double[2] *h)

    cdef void hash33(self, double[3] *p, double[3] *h)

    cdef double fade(self, double x)

    cdef double mix(self, double x, double y, double a)

    cdef double get_norm3(self, double[3] *v)

    cdef double get_norm2(self, double[2] *v)

    cdef unsigned int step(self, double a, double x)

    cdef double sign_with_abs(self, double *x)

    cdef (double, double) xy2pol(self, double x, double y)

    cdef double wrap2(self, double x, double y, bint rot=*)

    cpdef wrap(self, rot=*, t=*)




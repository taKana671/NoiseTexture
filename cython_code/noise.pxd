cdef class Noise:

    cdef:
        unsigned int[3] k
        unsigned int[3] u

    cdef void uhash33(self, unsigned int[3] *n)

    cdef double gtable3(self, unsigned int[3] *lattice, double[3] *p)

    cdef double fade(self, double x)

    cdef double mix(self, double x, double y, double a)





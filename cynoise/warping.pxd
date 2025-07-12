

cdef class Warping:

    cdef:
        double weight
        int octaves

cdef class DomainWarping2D(Warping):

    cdef:
        noise

    cdef double _warp2(self, double x, double y)

    cdef double _warp2_rot(self, double x, double y)

    cpdef double warp(self, double x, double y)

    cpdef double warp_rot(self, double x, double y)


cdef class DomainWarping3D(Warping):

    cdef:
        noise

    cdef double _warp3(self, double x, double y, double z)

    cpdef double warp(self, double x, double y, double z)


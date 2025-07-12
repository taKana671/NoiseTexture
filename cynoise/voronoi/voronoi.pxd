from ..noise cimport Noise


cdef class VoronoiNoise(Noise):

    cdef:
        int n_grid

    cdef void _vnoise2(self, double x, double y, double[2] *lattice_pt)

    cdef void _vnoise3(self, double x, double y, double z, double[3] *lattice_pt)

    cdef list _voronoi2(self, double x, double y)

    cdef list _voronoi3(self, double x, double y, double z)

    cpdef list voronoi2(self, double x, double y)

    cpdef list voronoi3(self, double x, double y, double z)

    cpdef noise2(self, size=*, t=*)

    cpdef noise2_color(self, size=*, cell=*, t=*)

    cpdef noise3(self, size=*, t=*)

    cpdef noise3_color(self, size=*, t=*)










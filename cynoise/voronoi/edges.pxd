from .voronoi cimport VoronoiNoise


cdef class VoronoiEdges(VoronoiNoise):

    cdef double _voronoi_edge2(self, double x, double y)

    cdef double _voronoi_edge3(self, double x, double y, double z)

    cpdef double voronoi_edge2(self, double x, double y)

    cpdef double voronoi_edge3(self, double x, double y, double z)

    cpdef double vmix1(self, double v, double cell=*, double edge=*)

    cpdef list vmix2(self, double x, double y, double cell=*, double edge=*)

    cpdef list vmix3(self, double x, double y, double z, double edge=*)

    cpdef noise2(self, size=*, cell=*, edge=*, t=*)

    cpdef noise2_color(self, size=*, cell=*, edge=*, t=*)

    cpdef noise3(self, size=*, cell=*, edge=*, t=*)

    cpdef noise3_color(self, size=*, edge=*, t=*)

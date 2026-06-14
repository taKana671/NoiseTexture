from .noise cimport Noise


cdef class PerlinNoise(Noise):

    cdef double _gtable2(self, double[2] *lattice, double[2] *p)

    cdef double _gtable3(self, double[3] *lattice, double[3] *p)

    cdef double _gtable4(self, double[4] *lattice, double[4] *p)

    cdef double _pnoise2(self, double x, double y)

    cdef double _pnoise3(self, double x, double y, double z)

    cdef double _pnoise4(self, double x, double y, double z, double w)

    cpdef double pnoise2(self, double x, double y)

    cpdef double pnoise3(self, double x, double y, double z)

    cpdef double pnoise4(self, double x, double y, double z, double w)

    cpdef noise2(self, size=*, grid=*, t=*)

    cpdef noise3(self, size=*, grid=*, t=*)

    cpdef noise4(self, size=*, grid=*, t=*)

    cpdef fractal2(self, size=*, grid=*, t=*, gain=*, lacunarity=*, octaves=*)

    cpdef warp2_rot(self, size=*, grid=*, t=*, weight=*, octaves=*)

    cpdef warp2(self, size=*, grid=*, t=*, octaves=*)
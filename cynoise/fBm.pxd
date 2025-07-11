

cdef class Fractal:

     cdef:
        double gain
        double lacunarity
        int octaves


cdef class Fractal2D(Fractal):

    cdef:
        noise

    cdef double _fractal2(self, double x, double y)
    
    cpdef double fractal(self, double x, double y)


cdef class Fractal3D(Fractal):

    cdef:
        noise

    cdef double _fractal3(self, double x, double y, double z)
    
    cpdef double fractal(self, double x, double y, double z)

    

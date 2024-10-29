from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        'noise',
        ['noise.pyx'],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        'perlin',
        ['perlin.pyx'],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        'fBm',
        ['fBm.pyx'],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        'cellular',
        ['cellular.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'voronoi',
        ['voronoi.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'periodic',
        ['periodic.pyx'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='noise_image',
    ext_modules=cythonize(
        extensions,
        compiler_directives={'profile': False}
    )
)
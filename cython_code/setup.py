from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


ext1 = Extension(
    'noise',
    ['noise.pyx'],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

ext2 = Extension(
    'perlin',
    ['perlin.pyx'],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

ext3 = Extension(
    'fBm',
    ['fBm.pyx'],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

ext4 = Extension(
    'cellular',
    ['cellular.pyx'],
    include_dirs=[numpy.get_include()]
)

ext5 = Extension(
    'voronoi',
    ['voronoi.pyx'],
    include_dirs=[numpy.get_include()]
)

ext6 = Extension(
    'periodic',
    ['periodic.pyx'],
    include_dirs=[numpy.get_include()]
)

setup(
    name='noise_image',
    ext_modules=cythonize([ext1, ext2, ext6])
)
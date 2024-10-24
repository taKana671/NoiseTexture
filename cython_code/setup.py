from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


ext1 = Extension(
    'noise',
    ['noise.pyx'],
    include_dirs=[numpy.get_include()]
)

ext2 = Extension(
    'perlin',
    ['perlin.pyx'],
    include_dirs=[numpy.get_include()]
)

setup(
    name='noise_image',
    ext_modules=cythonize([ext1, ext2])
)
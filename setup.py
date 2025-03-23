from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        'noise',
        ['cynoise/noise.pyx'],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        'perlin',
        ['cynoise/perlin.pyx'],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        'fBm',
        ['cynoise/fBm.pyx'],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        'cellular',
        ['cynoise/cellular.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'voronoi',
        ['cynoise/voronoi.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'periodic',
        ['cynoise/periodic.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'simplex',
        ['cynoise/simplex.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'value',
        ['cynoise/value.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'warping',
        ['cynoise/warping.pyx'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_package='cynoise',
    ext_modules=cythonize(
        extensions,
        compiler_directives={'profile': False}
    )
)
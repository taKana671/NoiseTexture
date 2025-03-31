from distutils.core import setup, Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        '*',
        ['cynoise/*.pyx'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'profile': False}
    )
)
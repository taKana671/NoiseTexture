import glob
import os
from distutils.core import setup, Extension
from Cython.Build import cythonize


def create_ext():
    for p in glob.iglob('cynoise/*pyx'):
        name = os.path.basename(p)
        yield Extension(
            f'cynoise.{name.split('.')[0]}',
            [f'cynoise/{name}'],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        )


extensions = [ext for ext in create_ext()]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'profile': False}
    )
)
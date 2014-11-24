from distutils.core import setup
from distutils.extension import Extension
import numpy as np

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }

if use_cython:
    ext_modules = [
        Extension("pomegranate.pomegranate", [ "pomegranate/base.pyx",
                                               "pomegranate/base.pxd",
                                               "pomegranate/bayesnet.pyx",
                                               "pomegranate/distributions.pxd",
                                               "pomegranate/fsm.pyx",
                                               "pomegranate/hmm.pyx",
                                               "pomegranate/utils.pxd" ], include_dirs=[np.get_include()]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules = [
        Extension("pomegranate.pomegranate", [ "pomegranate/base.c",
                                               "pomegranate/bayesnet.c",
                                               "pomegranate/distributions.c",
                                               "pomegranate/fsm.c",
                                               "pomegranate/hmm.c" ], include_dirs=[np.get_include()]),
    ]

setup(
    name='pomegranate',
    version='0.0.1',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['pomegranate'],
    url='http://pypi.python.org/pypi/pomegranate/',
    license='LICENSE.txt',
    description='Pomegranate is a graphical models library for Python, implemented in Cython for speed.',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=[
        "cython >= 0.20.1",
        "numpy >= 1.8.0",
        "scipy >= 0.13.3",
        "networkx >= 1.8.1",
        "matplotlib >= 1.3.1"
    ],
)

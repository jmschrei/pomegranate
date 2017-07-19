from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
    ext = 'c'
else:
    use_cython = True
    ext = 'pyx'

filenames = [ "base",
              "BayesianNetwork",
              "FactorGraph",
              "distributions",
              "hmm",
              "gmm",
              "NaiveBayes",
              "MarkovChain"
            ]

if not use_cython:
    extensions = [
        Extension( "pomegranate.{}".format( name ),
                   [ "pomegranate/{}.{}".format( name, ext ) ]) for name in filenames
    ]
else:
    extensions = [
            Extension("pomegranate.*", ["pomegranate/*.pyx"])
    ]

    extensions = cythonize( extensions )

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    name='pomegranate',
    version='0.7.7',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['pomegranate'],
    url='http://pypi.python.org/pypi/pomegranate/',
    license='LICENSE.txt',
    description='Pomegranate is a graphical models library for Python, implemented in Cython for speed.',
    ext_modules=extensions,
    cmdclass={'build_ext':build_ext},
    setup_requires=[
        "cython >= 0.22.1",
        "numpy >= 1.8.0",
        "scipy >= 0.17.0"
    ],
    install_requires=[
        "numpy >= 1.8.0",
        "joblib >= 0.9.0b4",
        "networkx >= 1.8.1",
        "scipy >= 0.17.0"
    ],
    test_suite = 'nose.collector',
    package_data={
        'pomegranate': ['*.pyd']
    },
    include_package_data=True,
)

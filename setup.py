from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
# https://stackoverflow.com/a/11181607/541202
# import __builtin__ as __builtins__

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
    ext = 'c'
else:
    use_cython = True
    ext = 'pyx'

filenames = [ 
    "base",
    "bayes",
    "BayesianNetwork",
    "MarkovNetwork",
    "FactorGraph",
    "hmm",
    "gmm",
    "kmeans",
    "NaiveBayes",
    "BayesClassifier",
    "MarkovChain",
    "utils",
    "parallel"
]

distributions = [
    'distributions',
    'UniformDistribution',
    'BernoulliDistribution',
    'NormalDistribution',
    'LogNormalDistribution',
    'ExponentialDistribution',
    'BetaDistribution',
    'GammaDistribution',
    'DiscreteDistribution',
    'PoissonDistribution',
    'KernelDensities',
    'IndependentComponentsDistribution',
    'MultivariateGaussianDistribution',
    'DirichletDistribution',
    'ConditionalProbabilityTable',
    'JointProbabilityTable'
]

if not use_cython:
    extensions = [
        Extension("pomegranate.{}".format( name ), [ "pomegranate/{}.{}".format(name, ext) ]) for name in filenames
    ] + [Extension("pomegranate.distributions.{}".format(dist), ["pomegranate/distributions/{}.{}".format(dist, ext)]) for dist in distributions]
else:
    extensions = [
            Extension("pomegranate.*", ["pomegranate/*.pyx"]),
	        Extension("pomegranate.distributions.*", ["pomegranate/distributions/*.pyx"])
    ]

    extensions = cythonize(extensions, compiler_directives={'language_level' : "2"})

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        if hasattr(__builtins__, '__NUMPY_SETUP__'):
            __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    name='pomegranate',
    version='0.14.9',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=[
        'pomegranate',
        'pomegranate/distributions',
    ],
    url='http://pypi.python.org/pypi/pomegranate/',
    license='MIT',
    description='Pomegranate is a graphical models library for Python, implemented in Cython for speed.',
    ext_modules=extensions,
    cmdclass={'build_ext':build_ext},
    setup_requires=[
        "cython >= 0.22.1, < 3.0.0",
        "numpy >= 1.20.0",
        "scipy >= 0.17.0"
    ],
    install_requires=[
        "cython >= 0.22.1, < 3.0.0",
        "numpy >= 1.20.0",
        "joblib >= 0.9.0b4",
        "networkx >= 2.4",
        "scipy >= 0.17.0",
        "pyyaml"
    ],
    extras_require={
        "Plotting": ["pygraphviz", "matplotlib"],
        "GPU": ["cupy"],
    },
    test_suite = 'nose.collector',
    package_data={
        'pomegranate': ['*.pyd', '*.pxd'],
        'pomegranate/distributions': ['*.pyd', '*.pxd'],
    },
    include_package_data=True,
    zip_safe=False,
)

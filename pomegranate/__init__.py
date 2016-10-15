# __init__.py: pomegranate
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )


"""
For detailed documentation and examples, see the README.
"""

import numpy as np
import os
import pyximport

# Adapted from Cython docs https://github.com/cython/cython/wiki/
# InstallingOnWindows#mingw--numpy--pyximport-at-runtime
if os.name == 'nt':
    if 'CPATH' in os.environ:
        os.environ['CPATH'] = os.environ['CPATH'] + np.get_include()
    else:
        os.environ['CPATH'] = np.get_include()

    # XXX: we're assuming that MinGW is installed in C:\MinGW (default)
    if 'PATH' in os.environ:
        os.environ['PATH'] = os.environ['PATH'] + ';C:\MinGW\bin'
    else:
        os.environ['PATH'] = 'C:\MinGW\bin'

    mingw_setup_args = { 'options': { 'build_ext': { 'compiler': 'mingw32' } }, 'include_dirs': np.get_include() }
    pyximport.install(setup_args=mingw_setup_args)

elif os.name == 'posix':
    if 'CFLAGS' in os.environ:
        os.environ['CFLAGS'] = os.environ['CFLAGS'] + ' -I' + np.get_include()
    else:
        os.environ['CFLAGS'] = ' -I' + np.get_include()

    pyximport.install()

from .hmm import *
from .kmeans import *
from .BayesianNetwork import *
from .FactorGraph import *
from .fsm import *
from .distributions import *
from .base import *
from .gmm import *
from .NaiveBayes import *
from .MarkovChain import *
from .parallel import *

__version__ = '0.6.1'

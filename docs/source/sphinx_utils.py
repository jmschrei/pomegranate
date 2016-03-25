# -*- coding: utf-8 -*-
"""Helper utilty function for customization."""
import sys
import os
import docutils
import subprocess

READTHEDOCS_BUILD = (os.environ.get('READTHEDOCS', None) is not None)

if not os.path.exists('../recommonmark'):
    subprocess.call('cd ..; rm -rf recommonmark;' +
                    'git clone https://github.com/tqchen/recommonmark', shell=True)
else:
    subprocess.call('cd ../recommonmark/; git pull', shell=True)

subprocess.call('pip install numpydoc', shell=True)
subprocess.call('pip install numpy', shell=True)
subprocess.call('pip install cython', shell=True)

sys.path.insert(0, os.path.abspath('../recommonmark/'))
sys.path.insert(0, "/".join( os.getcwd().split("/")[:-2] ))
sys.stderr.write('READTHEDOCS=%s\n' % (READTHEDOCS_BUILD))

from recommonmark import parser, transform

MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
# tools.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

'''
Tools taken from nose since it can no longer be installed after Py3.12.
'''


import re
import unittest


__all__ = ['ok_', 'eq_']

# Use the same flag as unittest itself to prevent descent into these functions:
__unittest = 1


def ok_(expr, msg=None):
    """Shorthand for assert. Saves 3 whole characters!
    """
    if not expr:
        raise AssertionError(msg)


def eq_(a, b, msg=None):
    """Shorthand for 'assert a == b, "%r != %r" % (a, b)
    """
    if not a == b:
        raise AssertionError(msg or "%r != %r" % (a, b))


#
# Expose assert* from unittest.TestCase
# - give them pep8 style names
#
caps = re.compile('([A-Z])')

def pep8(name):
    return caps.sub(lambda m: '_' + m.groups()[0].lower(), name)

class Dummy(unittest.TestCase):
    def nop():
        pass
_t = Dummy('nop')

for at in [ at for at in dir(_t)
            if at.startswith('assert') and not '_' in at ]:
    pepd = pep8(at)
    vars()[pepd] = getattr(_t, at)
    __all__.append(pepd)

del Dummy
del _t
del pep8

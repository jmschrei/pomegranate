"""
Copyright 2016 Oliver Schoenborn. BSD 3-Clause license (see __license__ at bottom of this file for details).

This module is part of the nose2pytest distribution.

This module's assert_ functions provide drop-in replacements for nose.tools.assert_ functions (many of which are
pep-8-ized extractions from Python's unittest.case.TestCase methods). As such, it can be imported in a test
suite run by pytest, to replace the nose imports with functions that rely on pytest's assertion
introspection for error reporting.  When combined with running nose2pytest.py on your test suite, this
module may be sufficient to decrease your test suite's third-party dependencies by 1.
"""

import pytest
import unittest


__all__ = [
    'assert_almost_equal',
    'assert_not_almost_equal',
    'assert_dict_contains_subset',

    'assert_raises_regex',
    'assert_raises_regexp',
    'assert_regexp_matches',
    'assert_warns_regex',
]


def assert_almost_equal(a, b, places=7, msg=None):
    """
    Fail if the two objects are unequal as determined by their
    difference rounded to the given number of decimal places
    and comparing to zero.

    Note that decimal places (from zero) are usually not the same
    as significant digits (measured from the most signficant digit).

    See the builtin round() function for places parameter.
    """
    if msg is None:
        assert round(abs(b - a), places) == 0
    else:
        assert round(abs(b - a), places) == 0, msg


def assert_not_almost_equal(a, b, places=7, msg=None):
    """
    Fail if the two objects are equal as determined by their
    difference rounded to the given number of decimal places
    and comparing to zero.

    Note that decimal places (from zero) are usually not the same
    as significant digits (measured from the most signficant digit).

    See the builtin round() function for places parameter.
    """
    if msg is None:
        assert round(abs(b - a), places) != 0
    else:
        assert round(abs(b - a), places) != 0, msg


def assert_dict_contains_subset(subset, dictionary, msg=None):
    """
    Checks whether dictionary is a superset of subset. If not, the assertion message will have useful details,
    unless msg is given, then msg is output.
    """
    dictionary = dictionary
    missing_keys = sorted(list(set(subset.keys()) - set(dictionary.keys())))
    mismatch_vals = {k: (subset[k], dictionary[k]) for k in subset if k in dictionary and subset[k] != dictionary[k]}
    if msg is None:
        assert missing_keys == [], 'Missing keys = {}'.format(missing_keys)
        assert mismatch_vals == {}, 'Mismatched values (s, d) = {}'.format(mismatch_vals)
    else:
        assert missing_keys == [], msg
        assert mismatch_vals == {}, msg


# make other unittest.TestCase methods available as-is as functions; trick taken from Nose

class _Dummy(unittest.TestCase):
    def do_nothing(self):
        pass

_t = _Dummy('do_nothing')

assert_raises_regex=_t.assertRaisesRegex,
assert_raises_regexp=_t.assertRaisesRegex,
assert_regexp_matches=_t.assertRegex,
assert_warns_regex=_t.assertWarnsRegex,

del _Dummy
del _t


# pytest integration: add all assert_ function to the pytest package namespace

# Use similar trick as Nose to bring in bound methods from unittest.TestCase as free functions:


def _supported_nose_name(name):
    return name.startswith('assert_') or name in ('ok_', 'eq_')


def pytest_configure():
    for name, obj in globals().items():
        if _supported_nose_name(name):
            setattr(pytest, name, obj)


# licensing

__license__ = """
    Copyright (c) 2016, Oliver Schoenborn
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of nose2pytest nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

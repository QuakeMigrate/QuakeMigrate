# -*- coding: utf-8 -*-
"""
Short test script for C onset functions.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import unittest

import numpy as np

from quakemigrate.core import overlapping_sta_lta, centred_sta_lta


class OnsetTests(unittest.TestCase):
    """Suite of tests to check the onset functions are working as expected."""

    def test_onsets_simple(self):
        """Confirm on simple case."""


        print("Testing onset functions behaviour on toy case.")
        signal = np.arange(6)
        nsta, nlta = 2, 3
        overlapping = overlapping_sta_lta(signal, nsta, nlta)
        expected = np.array([0, 0, 1.5, 39./28, 75./58, 123./100])
        print("\t1: Assert overlapping/classic STA/LTA onset correct...")
        self.assertTrue((expected == overlapping).all())

        centred = centred_sta_lta(signal, nsta, nlta)
        expected = np.array([0, 0, 7.5, 123/28, 0, 0])
        print("\t2: Assert centred STA/LTA onset correct...")
        self.assertTrue(np.allclose(centred, expected))
        print("\t   ...passed!")


if __name__ == "__main__":
    unittest.main()
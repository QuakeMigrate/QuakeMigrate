# -*- coding: utf-8 -*-
"""
Short test script for C onset functions.

"""

import pathlib
import unittest

import numpy as np
import obspy

from quakemigrate.core import overlapping_sta_lta, centred_sta_lta


test_dir = pathlib.Path.cwd()


class OnsetTests(unittest.TestCase):
    """Suite of tests to check the onset functions are working as expected."""

    def test_onsets_simple(self):
        """Confirm on simple case."""

        signal = np.arange(6)
        nsta, nlta = 2, 3
        overlapping = overlapping_sta_lta(signal, nsta, nlta)
        expected = np.array([0, 0, 1.5, 39./28, 75./58, 123./100])
        self.assertTrue((expected == overlapping).all())

        centred = centred_sta_lta(signal, nsta, nlta)
        expected = np.array([0, 0, 7.5, 123/28, 0, 0])
        self.assertTrue(np.allclose(centred, expected))

    def test_onsets_data(self):
        """Confirm on real data."""

        signal = obspy.read()[0].data
        ground_truth = obspy.read(str(test_dir / "test_data/onsets.m"))

        nsta, nlta = 10, 100
        overlapping = overlapping_sta_lta(signal, nsta, nlta)
        expected = ground_truth[0].data
        self.assertTrue((expected == overlapping).all())

        centred = centred_sta_lta(signal, nsta, nlta)
        expected = ground_truth[1].data
        self.assertTrue(np.allclose(centred, expected))


if __name__ == "__main__":
    unittest.main()

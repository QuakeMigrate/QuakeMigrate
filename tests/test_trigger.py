# -*- coding: utf-8 -*-
"""
Short test script for trigger functions (thresholds, smoothing, ...).

:copyright:
    2020–2024, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import unittest

import numpy as np
import pandas as pd

from quakemigrate import Trigger


class TriggerTests(unittest.TestCase):
    """Suite of tests to check the trigger functions are working as expected."""

    def test_trigger_thresholds(self):
        """Confirm on simple case."""

        print("\nTesting trigger threshold behaviour on toy case.")
        trigger = Trigger("not_a_lut", "test", "test")
        signal = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 4.0, 1.0])
        scandata = pd.Series(signal)

        trigger.threshold_method = "mad"
        trigger.mad_window_length = 4.0
        # Set multiplier to nullify scaling factor in util.caldculate_mad()
        trigger.mad_multiplier = 1 / 1.4826
        # Sampling rate of 1.0 (so window units = index units
        mad_threshold = trigger._get_threshold(scandata, 1.0)
        expected = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
        print("\t1: Assert mad trigger threshold correct...")
        self.assertTrue(np.allclose(mad_threshold, expected))
        print("\t   ...passed!")

        trigger.threshold_method = "median_ratio"
        trigger.median_window_length = 4.0
        trigger.median_multiplier = 2.0
        # Sampling rate of 1.0 (so window units = index units)
        median_ratio_threshold = trigger._get_threshold(scandata, 1.0)
        expected = np.array([2.0, 2.0, 2.0, 2.0, 9.0, 9.0, 9.0, 9.0])
        print("\t2: Assert median trigger threshold correct...")
        self.assertTrue(np.allclose(median_ratio_threshold, expected))
        print("\t   ...passed!")

    def test_smooth_coa(self):
        """Apply smoothing to simple example."""

        print("\nTesting trigger smoothing on toy case.")
        trigger = Trigger("not_a_lut", "test", "test")
        signal = np.array([1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0])
        data = pd.DataFrame({"COA": signal, "COA_N": signal})
        expected = np.array(
            [
                1.01826277,
                1.21596451,
                1.96788578,
                2.59577388,
                1.96788578,
                1.21596451,
                1.01826277,
            ]
        )
        # Using default smoothing_params (0.2 s sigma; 4-std kernel width)
        data = trigger._smooth_coa(data, 5.0)
        print("\t2: Assert smooth_coa output is correct...")
        self.assertTrue(np.allclose(data["COA"], expected))
        print("\t   ...passed!")


if __name__ == "__main__":
    unittest.main()

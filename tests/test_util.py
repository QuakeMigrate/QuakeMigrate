# -*- coding: utf-8 -*-
"""
Test script containing unit tests covering the functions contained in
quakemigrate.util.

:copyright:
    2020 - 2021, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import unittest
from copy import deepcopy

import numpy as np
from obspy import Trace, Stream, UTCDateTime

import quakemigrate.util as util


def mseed_stream():
    """Create a stream for testing."""

    rand = np.random.RandomState(815)
    header = {
        "network": "Z7",
        "station": "FLUR",
        "starttime": UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
        "npts": 412,
        "sampling_rate": 200.0,
        "channel": "HHE",
    }
    data = rand.randint(0, 1000, 412).astype(np.int32)
    trace1 = Trace(data=data, header=deepcopy(header))
    # Trace 2 = copy of Trace 1 with different dtype
    trace2 = trace1.copy()
    trace2.data = trace2.data.astype(float)
    # Trace 3 = trace with different channel (here different component)
    header["channel"] = "HHN"
    trace3 = Trace(data=data, header=deepcopy(header))

    return Stream(traces=[trace1, trace2, trace3])


class TestUtil(unittest.TestCase):
    """
    Suite of tests to verify behaviour of functions containedin the util
    sub-module of quakemigrate.

    """

    def test_merge_stream(self):
        """Test merging streams with different dtypes."""

        st = mseed_stream()
        st_merged = util.merge_stream(st)

        self.assertTrue(st_merged == Stream(st[2]))


if __name__ == "__main__":
    unittest.main()

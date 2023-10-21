# -*- coding: utf-8 -*-
"""
QuakeMigrate - a Python package for automatic earthquake detection and location
using waveform migration and stacking.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pkg_resources

import matplotlib
import os
import logging

from quakemigrate.io.data import Archive  # NOQA
from quakemigrate.lut import create_lut, read_nlloc, LUT  # NOQA
from quakemigrate.signal import QuakeScan, Trigger  # NOQA


# Set matplotlib logging level and backend
logging.getLogger("matplotlib").setLevel(logging.INFO)
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

__version__ = pkg_resources.get_distribution("quakemigrate").version

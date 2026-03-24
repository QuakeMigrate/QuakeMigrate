"""
QuakeMigrate - a Python package for automatic earthquake detection and location
using waveform migration and stacking.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging
import os
from importlib.metadata import version

import matplotlib

from quakemigrate.io.data import Archive
from quakemigrate.lut import create_lut, read_nlloc, LUT
from quakemigrate.signal import QuakeScan, Trigger


# Set matplotlib logging level and backend
logging.getLogger("matplotlib").setLevel(logging.INFO)
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

__all__ = ["Archive", "create_lut", "read_nlloc", "LUT", "QuakeScan", "Trigger"]
__version__ = version("quakemigrate")

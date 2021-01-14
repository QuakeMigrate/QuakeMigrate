# -*- coding: utf-8 -*-
"""

:copyright:
    2020 - 2021, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import matplotlib
import os
import logging

from quakemigrate.io.data import Archive  # NOQA
from quakemigrate.lut import create_lut, read_nlloc, LUT  # NOQA
from quakemigrate.signal import QuakeScan, Trigger  # NOQA

# Set matplotlib logging level and backend
logging.getLogger("matplotlib").setLevel(logging.INFO)
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")

name = "quakemigrate"
__version__ = "1.0.0"
__description__ = """QuakeMigrate - automatic earthquake detection and location
                     using waveform migration and stacking."""
__license__ = "GPLv3"
__author__ = "QuakeMigrate developers"
__email__ = """quakemigrate.developers@gmail.com, tom.winder@esc.cam.ac.uk,
               conor.bacon@esc.cam.ac.uk"""

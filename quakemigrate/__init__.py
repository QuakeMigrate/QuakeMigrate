# -*- coding: utf-8 -*-

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

name = "QuakeMigrate"
__version__ = "1.0.0"
__description__ = "QuakeMigrate - waveform backprojection for earthquake detection and location."
__license__ = "GPL v3.0"
__author__ = "QuakeMigrate developers"
__email__ = "jds70@cantab.net, tom.winder@esc.cam.ac.uk, conor.bacon@esc.cam.ac.uk"

# -*- coding: utf-8 -*-

import matplotlib
import logging

from QMigrate.io.data import Archive  # NOQA
from QMigrate.lut import create_lut, read_nlloc, LUT  # NOQA
from QMigrate.signal import QuakeScan, Trigger  # NOQA


logging.getLogger("matplotlib").setLevel(logging.INFO)


name = "QuakeMigrate"
__version__ = "1.0.0"
__description__ = "QuakeMigrate - waveform backprojection for earthquake detection and location."
__license__ = "GPL v3.0"
__author__ = "QuakeMigrate developers"
__email__ = "jds70@cantab.net, tom.winder@esc.cam.ac.uk, conor.bacon@esc.cam.ac.uk"

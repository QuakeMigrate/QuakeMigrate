# -*- coding: utf-8 -*-
"""
The :mod:`QMigrate.lut` module handles the definition and generation of the
traveltime lookup tables used in QuakeMigrate.

"""

from .create_lut import compute_traveltimes, read_nlloc  # NOQA
from .lut import LUT  # NOQA

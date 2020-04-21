# -*- coding: utf-8 -*-
"""
The :mod:`QMigrate.io` module handles the various input/output operations
performed by QuakeMigrate. This includes:

    * Reading waveform data - The submodule data.py can handle any waveform \
    data archives with regular directory structures.
    * Writing results - The submodule quakeio.py provides a suite of \
    functions to output QuakeMigrate results in the QuakeMigrate format.
    * Parse QuakeMigrate results into the ObsPy Catalog structure.
    * Various parsers to input files for different pieces of software. Feel \
    free to contribute more!

"""

from .quakeio import stations, read_vmodel, QuakeIO  # NOQA
from .data import Archive  # NOQA

# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.signal` module handles the core of the QuakeMigrate
methods. This includes:

    * Generation of onset functions from raw data.
    * Picking of waveforms from onset functions.
    * Raw scan for detect and locate.
    * Measurement of amplitudes and calculation of local earthquake magnitudes.

"""

from .scan import QuakeScan  # NOQA
from .trigger import Trigger  # NOQA

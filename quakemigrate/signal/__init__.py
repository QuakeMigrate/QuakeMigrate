# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.signal` module handles the core of the QuakeMigrate methods.
This includes:

    * Generation of onset functions from raw data.
    * Picking of waveforms from onset functions.
    * Migration of onsets for detect() and locate().
    * Measurement of phase amplitudes and calculation of local earthquake \
      magnitudes.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .scan import QuakeScan  # NOQA
from .trigger import Trigger  # NOQA

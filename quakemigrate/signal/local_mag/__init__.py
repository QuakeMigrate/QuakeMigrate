# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.local_mag` extension module handles the calculation of local
magnitudes from Wood-Anderson simulated waveforms.

.. warning:: The local_mag modules are an ongoing work in progress. We hope to\
 continue to extend their functionality, which may result in some API changes.\
 If you have comments or suggestions, please contact the QuakeMigrate \
developers at: quakemigrate.developers@gmail.com, or submit an issue on \
GitHub.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .local_mag import LocalMag  # NOQA
from .amplitude import Amplitude  # NOQA
from .magnitude import Magnitude  # NOQA

# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.onsets` module handles the generation of Onset functions. The
default method uses the ratio between the short-term and long-term averages of the
signal amplitude.

Feel free to contribute more Onset function options!

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .base import Onset  # NOQA
from .stalta import STALTAOnset, ClassicSTALTAOnset, CentredSTALTAOnset  # NOQA

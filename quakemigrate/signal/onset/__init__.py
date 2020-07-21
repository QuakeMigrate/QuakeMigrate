# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.onset` module handles the generation of Onset functions.
The default method uses the ratio between the short-term and long-term
averages.

Feel free to contribute more Onset function options!

:copyright:
    2020, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .onset import Onset  # NOQA
from .staltaonset import STALTAOnset, ClassicSTALTAOnset, CentredSTALTAOnset  # NOQA

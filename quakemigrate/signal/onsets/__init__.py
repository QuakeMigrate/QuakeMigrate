# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.onset` module handles the generation of Onset functions.
The default method uses the ratio between the short-term and long-term
averages.

Feel free to contribute more Onset function options!

"""

from .base import Onset  # NOQA
from .stalta import STALTAOnset, ClassicSTALTAOnset, CentredSTALTAOnset  # NOQA

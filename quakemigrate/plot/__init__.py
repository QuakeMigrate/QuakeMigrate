# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.plot` module provides methods for the generation of
figures in QuakeMigrate, including:
    * Event summaries
    * Phase pick summaries
    * Triggered event summaries
    * Amplitude / local magnitude summaries

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import matplotlib as mpl

from .event import event_summary  # NOQA
from .phase_picks import pick_summary  # NOQA
from .trigger import trigger_summary  # NOQA
from .amplitudes import amplitudes_summary  # NOQA


# Set the default colourmap
mpl.rc("image", cmap="viridis")

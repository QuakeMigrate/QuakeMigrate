"""
The :mod:`quakemigrate.plot` module provides methods for the generation of
figures in QuakeMigrate, including:
    * Event summaries
    * Phase pick summaries
    * Triggered event summaries
    * Amplitude / local magnitude summaries

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import matplotlib as mpl

from .amplitudes import amplitudes_summary
from .event import event_summary_2d, event_summary_3d
from .phase_picks import pick_summary
from .trigger import trigger_summary


# Set the default colourmap
mpl.rc("image", cmap="viridis")

__all__ = [
    "amplitudes_summary",
    "event_summary_2d",
    "event_summary_3d",
    "pick_summary",
    "trigger_summary",
]

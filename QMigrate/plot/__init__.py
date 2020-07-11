# -*- coding: utf-8 -*-
"""
The :mod:`QMigrate.plot` module provides methods for the generation of figures
in QuakeMigrate, including:
    * Event summaries
    * Phase pick summaries
    * Triggered event summaries

"""

import matplotlib.pyplot as plt

from .event import event_summary  # NOQA
from .phase_picks import pick_summary  # NOQA
from .trigger import trigger_summary  # NOQA
from .amplitudes import amplitudes_summary  # NOQA


plt.viridis()

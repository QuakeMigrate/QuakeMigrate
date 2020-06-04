# -*- coding: utf-8 -*-
"""
The :mod:`QMigrate.plot` module provides methods for the generation of figures
in QuakeMigrate, including:
    * Event summaries
    * Phase pick summaries
    * Triggered event summaries

"""

from .event import event_summary  # NOQA
from .phase_picks import pick_summary  # NOQA
from .trigger import trigger_summary  # NOQA

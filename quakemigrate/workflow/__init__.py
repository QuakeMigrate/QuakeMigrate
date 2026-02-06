"""
Workflow layer for QuakeMigrate.

This module contains a suite of tools for running each stage of QuakeMigrate from config
files.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .stages.lut import build
from .stages.detect import run as run_detect
from .stages.trigger import run as run_trigger
from .stages.locate import run as run_locate

__all__ = ["build", "run_detect", "run_trigger", "run_locate"]

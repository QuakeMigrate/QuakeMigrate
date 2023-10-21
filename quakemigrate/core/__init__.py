# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.core` module provides Python bindings for the library of
compiled C routines that form the core of QuakeMigrate:

    * Migrate onsets - This routine performs the continuous migration through \
    time and space of the onset functions. It has been parallelised with \
    openMP.
    * Find maximum coalescence - This routine finds the continuous maximum \
    coalescence amplitude in the 4-D coalesence volume.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .lib import migrate, find_max_coa  # NOQA

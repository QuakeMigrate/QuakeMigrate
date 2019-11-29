# -*- coding: utf-8 -*-
"""
Core
****

Wrapper module for the library of compiled C functions that perform the core
QuakeMigrate routines:

    * Migrate
    * Find maximum coalescence

"""

from .QMigratelib import migrate, find_max_coa

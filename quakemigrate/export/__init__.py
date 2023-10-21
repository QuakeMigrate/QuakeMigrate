# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.export` module provides some utility functions to export the
outputs of QuakeMigrate to other catalogue formats/software inputs:

    * Input files for NonLinLoc
    * ObsPy Catalog object
    * Snuffler pick/event files for manual phase picking
    * MFAST for shear-wave splitting analysis

.. warning:: Export modules are an ongoing work in progress. The functionality\
 of the core module `to_obspy` has been validated, but there may still be bugs\
 elsewhere. If you are interested in using these, or wish to add additional \
functionality, please contact the QuakeMigrate developers at: \
quakemigrate.developers@gmail.com .

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

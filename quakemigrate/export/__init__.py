# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.export` module provides some utility functions to export
the outputs of QuakeMigrate to other catalogue formats/software inputs:

    * Input files for NonLinLoc
    * ObsPy Catalog object
    * Snuffler pick/event files for manual phase picking
    * MFAST for shear-wave splitting analysis

.. warning:: Export modules are an ongoing work in progress. The functionality
of the core module `to_obspy` has been validated, but there may still be bugs
elsewhere. If you are interested in using these, or wish to add additional
functionality, please contact the QuakeMigrate developers at
quakemigrate.developers@gmail.com .

"""

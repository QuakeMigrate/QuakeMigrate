# -*- coding: utf-8 -*-
"""
The :mod:`quakemigrate.pick` module handles the picking of seismic phases. The
default method makes the phase picks by fitting a 1-D Gaussian to the Onset
function.

Feel free to contribute more phase picking methods!

"""

from .gaussian import GaussianPicker  # NOQA
from .base import PhasePicker  # NOQA

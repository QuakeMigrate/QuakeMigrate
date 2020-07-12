# -*- coding: utf-8 -*-
"""
The :mod:`QMigrate.pick` module handles the picking of seismic phases. The
default method makes the phase picks by fitting a 1-D Gaussian to the Onset
function.

Feel free to contribute more phase picking methods!

"""

from .gaussianpicker import GaussianPicker  # NOQA
from .pick import PhasePicker  # NOQA

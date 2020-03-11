# -*- coding: utf-8 -*-
"""
The :mod:`QMigrate.pick` module handles the picking of seismic phases. The
default method makes the phase picks by fitting a Gaussian to the Onset
function.

Feel free to contribute more methods of performing phase picking!

"""

from .pick import PhasePicker  # NOQA
from .gaussianpicker import GaussianPicker  # NOQA

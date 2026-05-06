"""
The :mod:`quakemigrate.plugins.pickers` module handles the picking of seismic phases.
The default method makes the phase picks by fitting a 1-D Gaussian to the Onset
function.

Feel free to contribute more phase picking methods!

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .base import PhasePicker
from .gaussian import build_gaussian_picker, GaussianPicker


__all__ = ["build_gaussian_picker", "GaussianPicker", "PhasePicker"]

"""
Small module that provides basic waveform simulations routines.

:copyright:
    2020–2024, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from .core import GaussianDerivativeWavelet, simulate_waveforms


__all__ = [GaussianDerivativeWavelet, simulate_waveforms]

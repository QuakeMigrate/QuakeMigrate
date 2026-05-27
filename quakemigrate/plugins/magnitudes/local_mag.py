"""
Module containing methods to calculate the local magnitude for an event located by
:mod:`QuakeMigrate`.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

import quakemigrate.util as util
from quakemigrate.io import write_amplitudes
from .amplitude import Amplitude, AmplitudeConfig
from .magnitude import Magnitude, MagnitudeConfig


if TYPE_CHECKING:
    from quakemigrate.io.core import Run
    from quakemigrate.io.event import Event
    from quakemigrate.lut import LUT


class LocalMag:
    """
    Plugin for calculating local magnitudes for located events.

    This plugin measures Wood-Anderson corrected waveform amplitudes, calculates
    per-trace local magnitude estimates, combines those estimates into a
    network-averaged local magnitude, and optionally writes an amplitude-vs-distance
    summary plot.

    Parameters
    ----------
    amplitude:
        Amplitude measurement configuration. May be an :class:`AmplitudeConfig` or a
        mapping accepted by :meth:`AmplitudeConfig.from_mapping`.
    magnitude:
        Magnitude calculation configuration. May be a :class:`MagnitudeConfig` or a
        mapping accepted by :meth:`MagnitudeConfig.from_mapping`.
    plot_amplitudes:
        Whether to write an amplitude-vs-distance summary plot for each event.

    Attributes
    ----------
    amp:
        Amplitude measurement helper.
    mag:
        Magnitude calculation helper.
    plot:
        Whether amplitude summary plots are written.

    See Also
    --------
    AmplitudeConfig
        Defines amplitude measurement options, defaults, and validation rules.
    MagnitudeConfig
        Defines magnitude calculation options, defaults, and validation rules.

    """

    stage: str = "locate_event"
    order: int = 350
    name: str = "LocalMagnitudes"
    kind: str = "magnitudes"

    def __init__(
        self,
        amplitude: AmplitudeConfig | Mapping[str, Any],
        magnitude: MagnitudeConfig | Mapping[str, Any],
        plot_amplitudes: bool = True,
    ) -> None:
        """Instantiate the LocalMag object."""

        self.amp = Amplitude(amplitude)
        self.mag = Magnitude(magnitude)
        self.plot = plot_amplitudes

    def __str__(self) -> str:
        """Return short summary string of the LocalMagnitudes object."""
        out = (
            "\tCalculating local magnitudes from Wood-Anderson corrected "
            "amplitude observations\n"
        )
        out += str(self.amp)
        out += str(self.mag)

        return out

    @util.timeit("info")
    def run(self, event: Event, lut: LUT, run: Run) -> Event:
        """
        Wrapper function to calculate the local magnitude of an event by first making
        Wood-Anderson corrected displacement amplitude measurements on each trace, then
        calculating magnitudes from these individual measurements, and a
        network-averaged (weighted) mean magnitude estimate and associated uncertainty.

        Additional functionality includes calculating an r^2 fit of the predicted
        amplitude with distance curve to the observed amplitudes, and an associated plot
        of amplitudes vs. distance.

        Parameters
        ----------
        event:
            Light class encapsulating waveform data, onset, pick and location
            information for a given event.
        lut:
            Contains the traveltime lookup tables for seismic phases, computed for some
            pre-defined velocity model.
        run:
            Light class encapsulating waveforms, coalescence information, picks and
            location information for a given event.

        Returns
        -------
        event:
            Light class encapsulating waveforms, coalescence information, picks and
            location information for a given event. Now also contains local magnitude
            information.

        """

        logging.info("\tCalculating magnitude...")

        # Measure amplitudes on all available traces
        amps = self.amp.get_amplitudes(event, lut)

        # Check if any amplitude measurements were made
        if amps[self.mag.amp_feature].isnull().all():
            logging.warning(
                "\t\tNo amplitude measurements were made! Skipping"
                " magnitude calculation"
            )
            write_amplitudes(run, amps, event)
            event.add_local_magnitude(np.nan, np.nan, np.nan)

            return event

        # Calculate magnitudes for individual amplitude measurements
        mags = self.mag.calculate_magnitudes(amps)

        # Write to file
        write_amplitudes(run, mags, event)

        # Combine magnitude estimates to calculate a network-averaged local
        # magnitude for the event. Optionally output a plot of amplitudes vs
        # distance.
        mag, mag_err, mag_r2, mags = self.mag.mean_magnitude(mags)

        event.add_local_magnitude(mag, mag_err, mag_r2)

        if self.plot and mag is not np.nan:
            self.mag.plot_amplitudes(
                mags, event, run, lut.unit_conversion_factor, self.amp.noise_measure
            )

        return event


def build_local_magnitude_plugin(
    amplitude: AmplitudeConfig | Mapping[str, Any],
    magnitude: MagnitudeConfig | Mapping[str, Any],
    plot_amplitudes: bool = True,
) -> LocalMag:
    """
    Build a :class:`LocalMag` plugin from grouped configuration values.

    This factory is intended for use by the plugin system. The amplitude and magnitude
    arguments correspond to the grouped TOML tables for the LocalMagnitudes plugin.

    Parameters
    ----------
    amplitude:
        Amplitude measurement configuration.
    magnitude:
        Magnitude calculation configuration.
    plot_amplitudes:
        Whether to write an amplitude-vs-distance summary plot for each event.

    Returns
    -------
    local_magnitude:
        Configured local magnitude plugin.

    """

    return LocalMag(
        amplitude=amplitude,
        magnitude=magnitude,
        plot_amplitudes=plot_amplitudes,
    )

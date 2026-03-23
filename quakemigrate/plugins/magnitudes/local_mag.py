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
from typing import TYPE_CHECKING

import numpy as np

import quakemigrate.util as util
from quakemigrate.io import write_amplitudes
from .amplitude import Amplitude
from .magnitude import Magnitude


if TYPE_CHECKING:
    from quakemigrate.io.core import Run
    from quakemigrate.io.event import Event
    from quakemigrate.lut import LUT


class LocalMag:
    """
    QuakeMigrate extension class for calculating local magnitudes.

    Provides functions for measuring amplitudes of earthquake waveforms and using these
    to calculate local magnitudes.

    Parameters
    ----------
    amp_params:
        All keys are optional, including:
        signal_window : float
            Length of S-wave signal window, in addition to the time window associated
            with the marginal_window and traveltime uncertainty.
        noise_window : float
            Length of the time window before the P-wave signal window in which to
            measure the noise amplitude.
        noise_measure : {"RMS", "STD", "ENV"}
            Method by which to measure the noise amplitude; root-mean-quare, standard
            deviation or average amplitude of the envelope of the signal.
        loc_method : {"spline", "gaussian", "covariance"}
            Which event location estimate to use.
        highpass_filter : bool
            Whether to apply a highpass filter to the data before measuring amplitudes.
        highpass_freq : float
            High-pass filter frequency. Required if highpass_filter is True.
        bandpass_filter : bool
            Whether to apply a band-pass filter before measuring amplitudes.
        bandpass_lowcut : float
            Band-pass filter low-cut frequency. Required if bandpass_filter is True.
        bandpass_highcut : float
            Band-pass filter high-cut frequency. Required if bandpass_filter is True.
        filter_corners : int
            Number of corners for the chosen filter.
        prominence_multiplier : float
            To set a prominence filter in the peak-finding algorithm.
            NOTE: not recommended for use in combination with a filter; filter
            gain corrections can lead to spurious results. Please see the
            `scipy.signal.find_peaks` documentation for further guidance.
    mag_params:
        Required keys:
        A0 : str or func
            Name of the attenuation function to use. Available options include
            {"Hutton-Boore", "keir2006", "UK", ...}. Alternatively specify a
            function which returns the attenuation factor at a specified
            (epicentral or hypocentral) distance.
        All other keys are optional, including:
        station_corrections : dict {str : float}
            Dictionary of trace_id : magnitude-correction pairs.
        amp_feature : {"S_amp", "P_amp"}
            Which phase amplitude measurement to use to calculate local magnitude.
        amp_multiplier : float
            Factor by which to multiply all measured amplitudes.
        use_hyp_dist : bool, optional
            Whether to use the hypocentral distance instead of the epicentral distance
            in the local magnitude calculation.
        trace_filter : regex expression
            Expression by which to select traces to use for the mean_magnitude
            calculation. E.g. '.*H[NE]$'.
        station_filter : list of str
            List of stations to exclude from the mean_magnitude calculation.
            E.g. ["KVE", "LIND"].
        dist_filter : float or False
            Whether to only use stations less than a specified (epicentral or
            hypocentral) distance from an event in the mean_magnitude() calculation.
            Distance in kilometres.
        pick_filter : bool
            Whether to only use stations where at least one phase was picked by the
            autopicker in the mean_magnitude calculation.
        noise_filter : float
            Factor by which to multiply the measured noise amplitude before excluding
            amplitude observations below the noise level.
        weighted_mean : bool
            Whether to do a weighted mean of the magnitudes when calculating the
            mean_magnitude.
    plot_amplitudes:
        Plot amplitudes vs. distance plot for each event.

    Attributes
    ----------
    amp : :class:`~quakemigrate.plugins.magnitudes.amplitude.Amplitude` object
        The Amplitude object for this instance of LocalMag. Contains functions
        to measure Wood-Anderson corrected displacement amplitudes for an event.
    mag : :class:`~quakemigrate.plugins.magnitudes.magnitude.Magnitude` object
        The Magnitude object for this instance of LocalMag. Contains functions to
        calculate magnitudes from Wood-Anderson corrected displacement amplitudes, and
        to combine them into a single magnitude estimate for the event.

    """

    def __init__(
        self, amp_params: dict, mag_params: dict, plot_amplitudes: bool = True
    ) -> None:
        """Instantiate the LocalMag object."""

        self.amp = Amplitude(amp_params)
        self.mag = Magnitude(mag_params)
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

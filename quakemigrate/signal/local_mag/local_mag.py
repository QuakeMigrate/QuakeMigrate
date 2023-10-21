# -*- coding: utf-8 -*-
"""
Module containing methods to calculate the local magnitude for an event located by
:mod:`QuakeMigrate`.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging
import numpy as np

import quakemigrate.util as util
from quakemigrate.io import write_amplitudes
from .amplitude import Amplitude
from .magnitude import Magnitude


class LocalMag:
    """
    QuakeMigrate extension class for calculating local magnitudes.

    Provides functions for measuring amplitudes of earthquake waveforms and using these
    to calculate local magnitudes.

    Parameters
    ----------
    amp_params : dict
        All keys are optional, including:
        signal_window : float
            Length of S-wave signal window, in addition to the time window associated
            with the marginal_window and traveltime uncertainty. (Default 0 s)
        noise_window : float
            Length of the time window before the P-wave signal window in which to
            measure the noise amplitude. (Default 10 s)
        noise_measure : {"RMS", "STD", "ENV"}
            Method by which to measure the noise amplitude; root-mean-quare, standard
            deviation or average amplitude of the envelope of the signal.
            (Default "RMS")
        loc_method : {"spline", "gaussian", "covariance"}
            Which event location estimate to use. (Default "spline")
        highpass_filter : bool
            Whether to apply a highpass filter to the data before measuring amplitudes.
            (Default False)
        highpass_freq : float
            High-pass filter frequency. Required if highpass_filter is True.
        bandpass_filter : bool
            Whether to apply a band-pass filter before measuring amplitudes.
            (Default: False)
        bandpass_lowcut : float
            Band-pass filter low-cut frequency. Required if bandpass_filter is True.
        bandpass_highcut : float
            Band-pass filter high-cut frequency. Required if bandpass_filter is True.
        filter_corners : int
            Number of corners for the chosen filter. Default: 4.
        prominence_multiplier : float
            To set a prominence filter in the peak-finding algorithm.
            (Default 0. = off).
            NOTE: not recommended for use in combination with a filter; filter
            gain corrections can lead to spurious results. Please see the
            `scipy.signal.find_peaks` documentation for further guidance.
    mag_params : dict
        Required keys:
        A0 : str or func
            Name of the attenuation function to use. Available options include
            {"Hutton-Boore", "keir2006", "UK", ...}. Alternatively specify a
            function which returns the attenuation factor at a specified
            (epicentral or hypocentral) distance. (Default "Hutton-Boore")
        All other keys are optional, including:
        station_corrections : dict {str : float}
            Dictionary of trace_id : magnitude-correction pairs. (Default None)
        amp_feature : {"S_amp", "P_amp"}
            Which phase amplitude measurement to use to calculate local magnitude.
            (Default "S_amp")
        amp_multiplier : float
            Factor by which to multiply all measured amplitudes.
        use_hyp_dist : bool, optional
            Whether to use the hypocentral distance instead of the epicentral distance
            in the local magnitude calculation. (Default False)
        trace_filter : regex expression
            Expression by which to select traces to use for the mean_magnitude
            calculation. E.g. '.*H[NE]$'. (Default None)
        station_filter : list of str
            List of stations to exclude from the mean_magnitude calculation.
            E.g. ["KVE", "LIND"]. (Default None)
        dist_filter : float or False
            Whether to only use stations less than a specified (epicentral or
            hypocentral) distance from an event in the mean_magnitude() calculation.
            Distance in kilometres. (Default False)
        pick_filter : bool
            Whether to only use stations where at least one phase was picked by the
            autopicker in the mean_magnitude calculation. (Default False)
        noise_filter : float
            Factor by which to multiply the measured noise amplitude before excluding
            amplitude observations below the noise level.
            (Default 1.)
        weighted_mean : bool
            Whether to do a weighted mean of the magnitudes when calculating the
            mean_magnitude. (Default False)
    plot_amplitudes : bool, optional
        Plot amplitudes vs. distance plot for each event. (Default True)

    Attributes
    ----------
    amp : :class:`~quakemigrate.signal.local_mag.amplitude.Amplitude` object
        The Amplitude object for this instance of LocalMag. Contains functions
        to measure Wood-Anderson corrected displacement amplitudes for an event.
    mag : :class:`~quakemigrate.signal.local_mag.magnitude.Magnitude` object
        The Magnitude object for this instance of LocalMag. Contains functions to
        calculate magnitudes from Wood-Anderson corrected displacement amplitudes, and
        to combine them into a single magnitude estimate for the event.

    Methods
    -------
    calc_magnitude(event, lut, run)

    """

    def __init__(self, amp_params, mag_params, plot_amplitudes=True):
        """Instantiate the LocalMag object."""

        self.amp = Amplitude(amp_params)
        self.mag = Magnitude(mag_params)
        self.plot = plot_amplitudes

    def __str__(self):
        """Return short summary string of the LocalMagnitudes object."""
        out = (
            "\tCalculating local magnitudes from Wood-Anderson corrected "
            "amplitude observations\n"
        )
        out += str(self.amp)
        out += str(self.mag)

        return out

    @util.timeit("info")
    def calc_magnitude(self, event, lut, run):
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
        event : :class:`~quakemigrate.io.event.Event` object
            Light class encapsulating waveform data, onset, pick and location
            information for a given event.
        lut : :class:`~quakemigrate.lut.lut.LUT` object
            Contains the traveltime lookup tables for seismic phases, computed for some
            pre-defined velocity model.
        run : :class:`~quakemigrate.io.core.Run` object
            Light class encapsulating waveforms, coalescence information, picks and
            location information for a given event.

        Returns
        -------
        event : :class:`~quakemigrate.io.event.Event` object
            Light class encapsulating waveforms, coalescence information, picks and
            location information for a given event. Now also contains local magnitude
            information.
        mag : float
            Network-averaged local magnitude estimate for this event.

        """

        # Measure amplitudes on all available traces
        amps = self.amp.get_amplitudes(event, lut)

        # Check if any amplitude measurements were made
        if amps[self.mag.amp_feature].isnull().all():
            logging.warning(
                "\t\tNo amplitude measurements were made! Skipping"
                " magnitude calculation"
            )
            write_amplitudes(run, amps, event)
            event.add_local_magnitude(np.nan, np.nan, np.nan, amps)

            return event, np.nan

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

        return event, mag

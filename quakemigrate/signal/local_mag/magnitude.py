# -*- coding: utf-8 -*-
"""
Module that supplies functions to calculate magnitudes from observations of trace
amplitudes, earthquake location, station locations, and an estimated attenuation curve
for the region of interest.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

from matplotlib import pyplot as plt
import numpy as np

from quakemigrate.plot.amplitudes import amplitudes_summary


class Magnitude:
    """
    Part of the QuakeMigrate LocalMag class; calculates local magnitudes from
    Wood-Anderson corrected waveform amplitude measurements.

    Takes waveform amplitude measurements from the LocalMag Amplitude class, and from
    these calculates local magnitude estimates using a local magnitude attenuation
    function. Magnitude corrections for individual stations and channels thereof can be
    applied, if provided.

    Individual estimates are then combined to calculate a network-averaged (weighted)
    mean local magnitude for the event. Also includes the function to measure the
    r-squared statistic assessing the goodness of fit between the predicted amplitude
    with distance from the nework-averaged local magnitude for the event and chosen
    attenuation function, and the observed amplitudes. This, provides a tool to
    distinguish between real microseismic events and artefacts.

    A summary plot illustrating the amplitude observations, their uncertainties, and the
    predicted amplitude with distance for the network-averaged local magnitude (and its
    uncertainties) can optionally be output.

    Attributes
    ----------
    A0 : str or func
        Name of the attenuation function to use. Available options include
        {"Hutton-Boore", "keir2006", "UK", ...}. Alternatively specify a function which
        returns the attenuation factor at a specified (epicentral or hypocentral)
        distance. (Default "Hutton-Boore")
    use_hyp_dist : bool, optional
        Whether to use the hypocentral distance instead of the epicentral distance in
        the local magnitude calculation. (Default False)
    amp_feature : {"S_amp", "P_amp"}
        Which phase amplitude measurement to use to calculate local magnitude.
        (Default "S_amp")
    station_corrections : dict {str : float}
        Dictionary of trace_id : magnitude-correction pairs. (Default None)
    amp_multiplier : float
        Factor by which to multiply all measured amplitudes.
    weighted_mean : bool
        Whether to use a weighted mean to calculate the network-averaged local magnitude
        estimate for the event. (Default False)
    trace_filter : regex expression
        Expression by which to select traces to use for the mean_magnitude calculation.
        E.g. ".*H[NE]$" . (Default None)
    noise_filter : float
        Factor by which to multiply the measured noise amplitude before excluding
        amplitude observations below the noise level.
        (Default 1.)
    station_filter : list of str
        List of stations to exclude from the mean_magnitude calculation.
        E.g. ["KVE", "LIND"]. (Default None)
    dist_filter : float or False
        Whether to only use stations less than a specified (epicentral or hypocentral)
        distance from an event in the mean_magnitude() calculation. Distance in
        kilometres. (Default False)
    pick_filter : bool
        Whether to only use stations where at least one phase was picked by the
        autopicker in the mean_magnitude calculation. (Default False)
    r2_only_used : bool
        Whether to only use amplitude observations which were used for the mean
        magnitude calculation when calculating the r-squared statistic for the goodness
        of fit between the measured and predicted amplitudes. Default: True; False is an
        experimental feature - use with caution.

    Methods
    -------
    calculate_magnitudes(amplitudes)
    mean_magnitude(magnitudes)
    plot_amplitudes(event, run)

    Raises
    ------
    TypeError
        If the user does not specify an A0 attenuation curve.
    ValueError
        If the user specifies an invalid A0 attenuation curve.

    """

    def __init__(self, magnitude_params={}):
        """Instantiate the Magnitude object."""

        # Parameters for individual magnitude calculation
        self.A0 = magnitude_params.get("A0")
        if not self.A0:
            raise TypeError("A0 attenuation correction not specified in params!")
        self.use_hyp_dist = magnitude_params.get("use_hyp_dist", False)
        self.amp_feature = magnitude_params.get("amp_feature", "S_amp")
        self.station_corrections = magnitude_params.get("station_corrections", {})
        self.amp_multiplier = magnitude_params.get("amp_multiplier", 1.0)

        # Parameters for mean magnitude calculation
        self.weighted_mean = magnitude_params.get("weighted_mean", False)
        self.trace_filter = magnitude_params.get("trace_filter")
        self.noise_filter = magnitude_params.get("noise_filter", 1.0)
        self.station_filter = magnitude_params.get("station_filter")
        self.dist_filter = magnitude_params.get("dist_filter", False)
        self.pick_filter = magnitude_params.get("pick_filter", False)
        self.r2_only_used = magnitude_params.get("r2_only_used", True)

    def __str__(self):
        """Return short summary string of the Magnitude object."""

        out = (
            "\t    Magnitude parameters:\n"
            f"\t\tA0 attenuation function = {self.A0}\n"
            f"\t\tUse hyp distance        = {self.use_hyp_dist}\n"
            f"\t\tAmplitude feature       = {self.amp_feature}\n"
        )
        if self.station_corrections:
            out += "\t\tUsing user-provided station corrections\n"
        out += (
            f"\t\tAmplitude multiplier    = {self.amp_multiplier}\n"
            f"\t\tUse weighted mean       = {self.weighted_mean}\n"
        )
        if self.trace_filter is not None:
            out += f"\t\tTrace filter            = {self.trace_filter}\n"
        out += f"\t\tNoise filter            = {self.noise_filter} x\n"
        if self.station_filter is not None:
            out += f"\t\tStation filter          = {self.station_filter}\n"
        if self.dist_filter:
            out += f"\t\tDistance filter         = {self.dist_filter} km\n"
        if self.pick_filter:
            out += "\t\tUsing only Amplitudes from picked traces.\n"

        return out

    def calculate_magnitudes(self, amplitudes):
        """
        Calculate magnitude estimates from amplitude measurements on individual stations
        /components.

        Parameters
        ----------
        amplitudes : `pandas.DataFrame` object
            P- and S-wave amplitude measurements for each component of each station in
            the look-up table.
            Columns:
                epi_dist : float
                    Epicentral distance between the station and the event hypocentre.
                z_dist : float
                    Vertical distance between the station and the event hypocentre.
                P_amp : float
                    Half maximum peak-to-trough amplitude in the P signal window. In
                    *millimetres*. Corrected for filter gain, if applicable.
                P_freq : float
                    Approximate frequency of the maximum amplitude P-wave signal.
                    Calculated from the peak-to-trough time interval of the max
                    peak-to-trough amplitude.
                P_time : `obspy.UTCDateTime` object
                    Approximate time of amplitude observation (halfway between peak and
                    trough times).
                P_avg_amp : float
                    Average amplitude in the P signal window, measured by the same
                    method as the Noise_amp (see `noise_measure`) and corrected for the
                    same filter gain as `P_amp`. In *millimetres*.
                P_filter_gain : float or NaN
                    Filter gain at `P_freq` - which has been corrected for in the P_amp
                    measurements - if a filter was applied prior to amplitude
                    measurement; Else NaN.
                S_amp : float
                    As for P, but in the S wave signal window.
                S_freq : float
                    As for P.
                S_time : `obspy.UTCDateTime` object
                    As for P.
                S_avg_amp : float
                    As for P.
                S_filter_gain : float or NaN.
                    As for P.
                Noise_amp : float
                    The average signal amplitude in the noise window. In *millimetres*.
                    See `noise_measure` parameter.
                is_picked : bool
                    Whether at least one of the phase arrivals was picked by the
                    autopicker.
            Index = Trace ID (see `obspy.Trace` object property 'id')

        Returns
        -------
        magnitudes : `pandas.DataFrame` object
            The original amplitudes DataFrame, with columns containing the calculated
            magnitude and an associated error now added.
            Columns = ["epi_dist", "z_dist", "P_amp", "P_freq", "P_time",
                   "P_avg_amp", "P_filter_gain", "S_amp", "S_freq", "S_time",
                   "S_avg_amp", "S_filter_gain", "Noise_amp", "is_picked",
                   "ML", "ML_Err"]
            Index = Trace ID (see `obspy.Trace.id`)
            Additional fields:
            ML : float
                Magnitude calculated from the chosen amplitude measurement, using the
                specified attenuation curve and station_corrections.
            ML_Err : float
                Estimate of the error on the calculated magnitude, based on potential
                errors in the maximum amplitude measurement according to the measured
                noise amplitude.

        Raises
        ------
        AttributeError
            If A0 attenuation correction is not specified.

        """

        trace_ids = amplitudes.index
        amps = amplitudes[self.amp_feature].values * self.amp_multiplier
        noise_amps = amplitudes["Noise_amp"].values * self.amp_multiplier
        filter_gains = amplitudes[f"{self.amp_feature[0]}_filter_gain"]
        if not filter_gains.isnull().values.any():
            noise_amps /= filter_gains

        # Remove those amplitudes where the noise is greater than the amplitude and set
        # amplitudes which = 0. to NaN (to avoid logs blowing up).
        with np.errstate(invalid="ignore"):
            amps[amps < noise_amps] = np.nan
            amps[amps == 0.0] = np.nan

        # Calculate distances (hypocentral or epicentral)
        edist, zdist = amplitudes["epi_dist"], amplitudes["z_dist"]
        if self.use_hyp_dist:
            dist = np.sqrt(edist.values**2 + zdist.values**2)
        else:
            dist = edist.values
        dist[dist == 0.0] = np.nan

        # Calculate magnitudes and associated errors
        mags, mag_errs = self._calc_mags(trace_ids, amps, noise_amps, dist)

        magnitudes = amplitudes.copy()
        magnitudes["ML"] = mags
        magnitudes["ML_Err"] = mag_errs

        return magnitudes

    def mean_magnitude(self, magnitudes):
        """
        Calculate the network-averaged local magnitude for an event based on the
        magnitude estimates calculated from amplitude measurements made on each
        component of each station.

        The user may specify distance, station, channel and a number of other filters to
        restrict which observations are included in this best estimate of the local
        magnitude of the event.

        Parameters
        ----------
        magnitudes : `pandas.DataFrame` object
            Contains P- and S-wave amplitude measurements for each component of each
            station in the look-up table, and local magnitude estimates calculated from
            them (output by calculate_magnitudes()). Note that the amplitude
            observations are raw, but the ML estimates derived from them include station
            corrections, if provided.
            Columns:
                epi_dist : float
                    Epicentral distance between the station and the event hypocentre.
                z_dist : float
                    Vertical distance between the station and the event hypocentre.
                P_amp : float
                    Half maximum peak-to-trough amplitude in the P signal window. In
                    *millimetres*. Corrected for filter gain, if applicable.
                P_freq : float
                    Approximate frequency of the maximum amplitude P-wave signal.
                    Calculated from the peak-to-trough time interval of the max
                    peak-to-trough amplitude.
                P_time : `obspy.UTCDateTime` object
                    Approximate time of amplitude observation (halfway between peak and
                    trough times).
                P_avg_amp : float
                    Average amplitude in the P signal window, measured by the same
                    method as the Noise_amp (see `noise_measure`) and corrected for the
                    same filter gain as `P_amp`. In *millimetres*.
                P_filter_gain : float or NaN
                    Filter gain at `P_freq` - which has been corrected for in the P_amp
                    measurements - if a filter was applied prior to amplitude
                    measurement; Else NaN.
                S_amp : float
                    As for P, but in the S wave signal window.
                S_freq : float
                    As for P.
                S_time : `obspy.UTCDateTime` object
                    As for P.
                S_avg_amp : float
                    As for P.
                S_filter_gain : float or NaN.
                    As for P.
                Noise_amp : float
                    The average signal amplitude in the noise window. In *millimetres*.
                    See `noise_measure` parameter.
                is_picked : bool
                    Whether at least one of the phase arrivals was picked by the
                    autopicker.
                ML : float
                    Magnitude calculated from the chosen amplitude measurement, using
                    the specified attenuation curve and station_corrections.
                ML_Err : float
                    Estimate of the error on the calculated magnitude, based on
                    potential errors in the maximum amplitude measurement according to
                    the measured noise amplitude.
            Index = Trace ID (see `obspy.Trace` object property 'id')

        Returns
        -------
        mean_mag : float or NaN
            Network-averaged local magnitude estimate for the event. Mean (or weighted
            mean) of the magnitude estimates calculated from each individual channel
            after optionally removing some observations based on trace ID, distance,
            etcetera.
        mean_mag_err : float or NaN
            Standard deviation (or weighted standard deviation) of the magnitude
            estimates calculated from individual channels which contributed to the
            calculation of the (weighted) mean magnitude.
        mag_r_squared : float or NaN
            r-squared statistic describing the fit of the amplitude vs. distance curve
            predicted by the calculated mean_mag and chosen attenuation model to the
            measured amplitude observations. This is intended to be used to help
            discriminate between 'real' events, for which the predicted amplitude vs.
            distance curve should provide a good fit to the observations, from
            artefacts, which in general will not.

        """

        # Get station corrections
        corrs = [
            self.station_corrections[t] if t in self.station_corrections.keys() else 0.0
            for t in magnitudes.index
        ]
        magnitudes["Station_Correction"] = corrs

        # Correct noise amps for filter gain, if applicable
        filter_gains = magnitudes[f"{self.amp_feature[0]}_filter_gain"]
        if not filter_gains.isnull().values.any():
            magnitudes.loc[:, "Noise_amp"] /= filter_gains

        # Do filtering
        used_mags, all_mags = self._filter_mags(magnitudes)

        # Check if there are still some magnitude observations left
        if len(used_mags) == 0:
            logging.warning(
                "\t    No magnitude observations match the "
                "filtering criteria! Skipping."
            )
            return np.nan, np.nan, np.nan, all_mags

        mags = used_mags["ML"].values

        # If weighted, calculate weight as (1/error)^2. Else equal weighting.
        if self.weighted_mean:
            weights = (1 / used_mags["ML_Err"]) ** 2
        else:
            weights = np.ones_like(mags)

        # Calculate mean and standard deviation. NOTE: makes the assumption that the
        # distribution of these magnitude observations can locally be approximated by a
        # normal distribution. In reality it will have a negative skew, making the mean
        # magnitude a slight underestimate.
        mean_mag = np.sum(mags * weights) / np.sum(weights)
        mean_mag_err = np.sqrt(
            np.sum(((mags - mean_mag) * weights) ** 2) / np.sum(weights)
        )

        # Pass the magnitudes (filtered & un-filtered) to the _mag_r_squared
        # function.
        mag_r_squared = self._mag_r_squared(
            all_mags, mean_mag, only_used=self.r2_only_used
        )

        return mean_mag, mean_mag_err, mag_r_squared, all_mags

    def plot_amplitudes(
        self, magnitudes, event, run, unit_conversion_factor, noise_measure="RMS"
    ):
        """
        Plot a figure showing the measured amplitude with distance vs. predicted
        amplitude with distance derived from mean_mag and the chosen attenuation model.

        The amplitude observations (both for noise and signal amplitudes) are corrected
        according to the same station corrections that were used in calculating the
        individual local magnitude estimates that were used to calculate the
        network-averaged local magnitude for the event.

        Parameters
        ----------
        magnitudes : `pandas.DataFrame` object
            Contains P- and S-wave amplitude measurements for each component of each
            station in the look-up table, and local magnitude estimates calculated from
            them (output by calculate_magnitudes()). Note that the amplitude
            observations are raw, but the ML estimates derived from them include station
            corrections, if provided.
            Columns = ["epi_dist", "z_dist", "P_amp", "P_freq", "P_time", "P_avg_amp",
                       "P_filter_gain", "S_amp", "S_freq", "S_time", "S_avg_amp",
                       "S_filter_gain", "Noise_amp", "is_picked", "ML", "ML_Err",
                       "Noise_Filter", "Trace_Filter", "Station_Filter", "Dist_Filter",
                       "Dist", "Used"]
        event : :class:`~quakemigrate.io.event.Event` object
            Light class encapsulating waveforms, coalescence information, picks and
            location information for a given event.
        run : :class:`~quakemigrate.io.core.Run` object
            Light class encapsulating i/o path information for a given run.
        unit_conversion_factor : float
            A conversion factor based on the lookup table grid projection, used to
            ensure the location uncertainties have units of kilometres.

        """

        mag = event.localmag["ML"]
        mag_err = event.localmag["ML_Err"]
        mag_r2 = event.localmag["ML_r2"]

        # For amplitudes and magnitude calculation, distances must be in km
        km_cf = 1000 / unit_conversion_factor

        # Calculate distance error (for errorbars - using gaussian uncertainty)
        x_err, y_err, z_err = event.get_loc_uncertainty("gaussian") / km_cf
        epi_err = np.sqrt(x_err**2 + y_err**2)

        if self.use_hyp_dist:
            dist_err = np.sqrt(epi_err**2 + z_err**2)
        else:
            dist_err = epi_err

        all_amps = (
            magnitudes[self.amp_feature].values
            * self.amp_multiplier
            * np.power(10, magnitudes["Station_Correction"])
        )
        noise_amps = (
            magnitudes["Noise_amp"].values
            * self.amp_multiplier
            * np.power(10, magnitudes["Station_Correction"])
        )
        filter_gains = magnitudes[f"{self.amp_feature[0]}_filter_gain"]
        if not filter_gains.isnull().values.any():
            noise_amps /= filter_gains

        dist = magnitudes["Dist"]

        # Find min/max values for x and y axes
        amps_max = all_amps.max() * 5
        amps_min = noise_amps.min() / 10
        dist_min = dist.min() / 2
        dist_max = dist.max() * 1.5

        _, ax = amplitudes_summary(
            magnitudes,
            self.amp_feature,
            self.amp_multiplier,
            dist_err,
            mag_r2,
            noise_measure,
        )

        # -- Calculate predicted amplitudes from ML & attenuation function --
        # Upper and lower bounds for predicted amplitude from upper/lower bounds for mag
        mag_upper = mag + mag_err
        mag_lower = mag - mag_err

        # Calculated attenuation correction for full range of distances
        distances = np.linspace(dist_min, dist_max, 10000)
        att = self._get_attenuation(distances)

        # Calculate predicted amplitude with distance
        predicted_amp = np.power(10, (mag - att))
        predicted_amp_upper = np.power(10, (mag_upper - att))
        predicted_amp_lower = np.power(10, (mag_lower - att))

        # Plot predicted amplitude with distance
        label = (
            f"Predicted amplitude for ML = {mag:.2f} \u00B1 {mag_err:.2f}"
            f'\nusing attenuation curve "{self.A0}"'
        )
        ax.plot(distances, predicted_amp, linestyle="-", c="r", label=label)
        ax.plot(distances, predicted_amp_upper, linestyle="--", c="r")
        ax.plot(distances, predicted_amp_lower, linestyle="--", c="r")

        # If distance filter specified, add it to the plot
        if self.dist_filter:
            ax.axvline(
                self.dist_filter,
                linestyle="--",
                ymin=0,
                ymax=amps_max,
                color="k",
                label="Distance filter",
            )

        # Set axis limits
        ax.set_xlim(dist_min, dist_max)
        ax.set_ylim(amps_min, max(np.max(predicted_amp), amps_max))

        # Set figure and axis titles
        ax.set_title(
            f'Amplitude vs distance plot for event: "{event.uid}"', fontsize=18
        )
        ax.set_ylabel("Amplitude / mm", fontsize=16)
        if self.use_hyp_dist:
            ax.set_xlabel("Hypocentral Distance / km", fontsize=16)
        else:
            ax.set_xlabel("Epicentral Distance / km", fontsize=16)

        # Add legend
        ax.legend(fontsize=16, loc="upper right")

        # Specify tight layout
        plt.tight_layout()

        fpath = run.path / "locate" / run.subname / "amplitude_plots"
        fpath.mkdir(exist_ok=True, parents=True)
        fstem = f"{run.name}_{event.uid}_AmpVsDistance"
        file = (fpath / fstem).with_suffix(".pdf")
        plt.savefig(file, dpi=400)
        plt.close("all")

    def _calc_mags(self, trace_ids, amps, noise_amps, dist):
        """
        Calculates magnitudes from a series of amplitude measurements.

        Parameters
        ----------
        trace_ids : array-like, contains strings
            List of ID strings for each trace.
        amps : array-like, contains floats
            Measurements of *half* peak-to-trough amplitudes, in *millimetres*
        noise_amps : array-like, contains floats
            Estimate of uncertainty in amplitude measurements caused by noise on the
            signal. Also in mm.
        dist : array-like, contains floats
            Distances between source and receiver in kilometres.

        Returns
        -------
        mags : array-like
            Magnitudes for each channel calculated from the chosen amplitude measurement
            (P or S).
        mag_errs : array-like
            Estimate of the error on the calculated magnitude, based on potential errors
            in the maximum amplitude measurement according to the measured noise
            amplitude.

        """

        # Read in station corrections for each trace
        corrs = [
            self.station_corrections[t] if t in self.station_corrections.keys() else 0.0
            for t in trace_ids
        ]

        att = self._get_attenuation(dist)

        # Calculate magnitudes
        mags = np.log10(amps) + att + np.array(corrs)

        # Simple estimate of magnitude error based on the upper and lower bounds of the
        # amplitude measurements according to the measured noise amplitude
        upper_mags = np.log10(amps + noise_amps) + att + np.array(corrs)
        lower_mags = np.log10(amps - noise_amps) + att + np.array(corrs)
        mag_errs = upper_mags - lower_mags

        return mags, mag_errs

    def _get_attenuation(self, dist):
        """
        Calculate attenuation according to user-provided or built-in logA0 attenuation
        function.

        Parameters
        ----------
        dist : float or array-like
            Distance(s) between source and receiver.

        Returns
        -------
        att : float or array-like
            Attenuation correction factor.

        """

        if callable(self.A0):
            att = self.A0(dist)
        else:
            att = self._logA0(dist)

        return att

    def _logA0(self, dist):
        """
        A set of logA0 attenuation correction equations from the literature.
        Feel free to add more.

        Currently implemented:
            "Hutton-Boore" : Southern California (Hutton & Boore, 1987)
            "keir2006" : Ethiopia (Keir et al., 2006)
            "Danakil2017" : Illsley-Kemp et al. (2017) - Danakil Depression,
                            Afar
            "Greenfield2018" : Northern Volcanic Zone, Iceland (Greenfield
                               et al., 2018)
            "Greenfield2018_askja" : Askja, Iceland (Greenfield et al., 2018)
            "Greenfield2018_bardarbunga" : Bardarbunga, Iceland (Greenfield et
                                           al., 2018)
            "langston1998" : Tanzania, East Africa (Langston, 1998)
            "UK" : UK (Luckett et al., 2018)

        Parameters
        ----------
        dist : float
            Distance between source and receiver.

        Returns
        -------
        logA0 : float
            Attenuation correction factor.

        Raises
        ------
        ValueError
            If invalid A0 attenuation function is specified.

        """

        eqn = self.A0

        if eqn == "keir2006":
            att = 1.196997 * np.log10(dist / 17.0) + 0.001066 * (dist - 17.0) + 2.0
        elif eqn == "Danakil2017":
            att = 1.274336 * np.log10(dist / 17.0) - 0.000273 * (dist - 17.0) + 2.0
        elif eqn == "Greenfield2018_askja":
            att = 1.4406 * np.log10(dist / 17.0) + 0.003 * (dist - 17.0) + 2.0
        elif eqn == "Greenfield2018_bardarbunga":
            att = 1.2534 * np.log10(dist / 17.0) + 0.0032 * (dist - 17.0) + 2.0
        elif eqn == "Greenfield2018_comb":
            att = 1.1999 * np.log10(dist / 17.0) + 0.0016 * (dist - 17.0) + 2.0
        elif eqn == "Hutton-Boore":
            att = 1.11 * np.log10(dist / 100.0) + 0.00189 * (dist - 100.0) + 3.0
        elif eqn == "Langston1998":
            att = 0.776 * np.log10(dist / 17.0) + 0.000902 * (dist - 17) + 2.0
        elif eqn == "UK":
            att = (
                1.11 * np.log10(dist)
                + 0.00189 * dist
                - 1.16 * np.exp(-0.2 * dist)
                - 2.09
            )
        else:
            raise ValueError(eqn, "is not a valid A0 attenuation function.")

        return att

    def _filter_mags(self, magnitudes):
        """
        Filter magnitudes observations according to the user-specified filters for
        source-station distance, trace ID, etc.

        Parameters
        ----------
        magnitudes : `pandas.DataFrame` object
            Contains P- and S-wave amplitude measurements for each component of each
            station in the look-up table, and local magnitude estimates calculated from
            them (output by calculate_magnitudes()). Note that the amplitude
            observations are raw, but the ML estimates derived from them include station
            corrections, if provided.
            Columns = ["epi_dist", "z_dist", "P_amp", "P_freq", "P_time", "P_avg_amp",
                       "P_filter_gain", "S_amp", "S_freq", "S_time", "S_avg_amp",
                       "S_filter_gain", "Noise_amp", "is_picked", "ML", "ML_Err"]

        Returns
        -------
        used_mags : `pandas.DataFrame` object
            As input, but only including individual amplitude measurements / local
            magnitude estimates that meet the filters specified by the user. Now with
            additional columns:
            Noise_Filter : bool
                Whether this observation meets the noise filter.
            Trace_Filter : bool
                Whether this observation matches the trace filter.
            Station_Filter : bool
                Whether this observation is not excluded by the station filter.
            Dist_Filter : bool
                Whether this observation meets the distance filter.
            Dist : bool
                The (epicentral or hypocentral) distance between the station and event.
            Used : bool (== True)
                Whether the observation meets all filter requirements.
        all_mags : `pandas.DataFrame` object
            As for used_mags, but containing all observations from the input magnitudes
            DataFrame, other than those which feature null values for the signal or
            noise amplitude.

        """

        # Remove nan amplitude values
        magnitudes.dropna(subset=[self.amp_feature, "Noise_amp"], inplace=True)

        # Apply noise filter.
        if self.noise_filter != 0.0:
            amps = magnitudes[self.amp_feature].values
            noise_amps = magnitudes["Noise_amp"].values
            magnitudes["Noise_Filter"] = False
            with np.errstate(invalid="ignore"):
                magnitudes.loc[
                    (amps > noise_amps * self.noise_filter), "Noise_Filter"
                ] = True

        # Apply trace filter
        if self.trace_filter is not None:
            magnitudes["Trace_Filter"] = False
            magnitudes.loc[
                magnitudes.index.str.contains(self.trace_filter), "Trace_Filter"
            ] = True

        # Apply station filter
        if self.station_filter is not None:
            magnitudes["Station_Filter"] = True
            for stn in list(self.station_filter):
                magnitudes.loc[
                    magnitudes.index.str.contains(f".{stn}.", regex=False),
                    "Station_Filter",
                ] = False

        # Calculate distances
        edist, zdist = magnitudes["epi_dist"], magnitudes["z_dist"]
        if self.use_hyp_dist:
            dist = np.sqrt(edist.values**2 + zdist.values**2)
        else:
            dist = edist.values

        # Apply distance filter
        if self.dist_filter:
            magnitudes["Dist_Filter"] = False
            magnitudes.loc[(dist <= self.dist_filter), "Dist_Filter"] = True

        # Set distances; remove dist=0 values (logs do not like this)
        dist[dist == 0.0] = np.nan
        magnitudes["Dist"] = dist

        # Identify used mags (after applying all filters)
        magnitudes["Used"] = True
        if self.trace_filter is not None:
            magnitudes.loc[~magnitudes["Trace_Filter"], "Used"] = False
        if self.station_filter is not None:
            magnitudes.loc[~magnitudes["Station_Filter"], "Used"] = False
        if self.dist_filter:
            magnitudes.loc[~magnitudes["Dist_Filter"], "Used"] = False
        if self.pick_filter:
            magnitudes.loc[~magnitudes["is_picked"], "Used"] = False
        if self.noise_filter != 0.0:
            magnitudes.loc[~magnitudes["Noise_Filter"], "Used"] = False

        used_mags = magnitudes[magnitudes["Used"]]

        return used_mags, magnitudes

    def _mag_r_squared(self, magnitudes, mean_mag, only_used=True):
        """
        Calculate the r-squared statistic for the fit of the amplitudes vs distance
        model predicted by the estimated event magnitude and the chosen attenuation
        function to the observed amplitudes. The fit is calculated in log(amplitude)
        space to linearise the data, in order for the calculation of the r-squared
        statistic to be appropriate.

        The raw amplitude observations are corrected according to the station
        corrections that were used in calculating the local magnitudes from which the
        network-averaged local magnitude was calculated.

        Parameters
        ----------
        magnitudes : `pandas.DataFrame` object
            Contains P- and S-wave amplitude measurements for each component of each
            station in the look-up table, and local magnitude estimates calculated from
            them (output by calculate_magnitudes()). Note that the amplitude
            observations are raw, but the ML estimates derived from them include station
            corrections, if provided.
            Columns = ["epi_dist", "z_dist", "P_amp", "P_freq", "P_time", "P_avg_amp",
                       "P_filter_gain", "S_amp", "S_freq", "S_time", "S_avg_amp",
                       "S_filter_gain", "Noise_amp", "is_picked", "ML", "ML_Err",
                       "Noise_Filter", "Trace_Filter", "Station_Filter", "Dist_Filter",
                       "Dist", "Used"]
        mean_mag : float or NaN
            Network-averaged local magnitude estimate for the event. Mean (or weighted
            mean) of the magnitude estimates calculated from each individual channel
            after optionally removing some observations based on trace ID, distance,
            etcetera.
        only_used : bool
            Only calculate the r-squared value from those magnitudes which were included
            in calculating the network-averaged `mean_mag`.

        Returns
        -------
        mag_r_squared : float or NaN
            r-squared statistic describing the fit of the amplitude vs. distance curve
            predicted by the calculated mean_mag and chosen attenuation model to the
            measured amplitude observations. This is intended to be used to help
            discriminate between 'real' events, for which the predicted amplitude vs.
            distance curve should provide a good fit to the observations, from
            artefacts, which in general will not.

        Raises
        ------
        AttributeError
            If the user selects `only_used=False` but does not specify a noise filter.

        """

        if only_used:
            # Only keep magnitude estimates which meet all the user-specified filter
            # requirements.
            magnitudes = magnitudes[magnitudes["Used"]]
        else:
            # Apply a default set of filters (including some of the user-specified
            # filters)
            if self.trace_filter is not None:
                magnitudes = magnitudes[magnitudes["Trace_Filter"]]
            if self.station_filter is not None:
                magnitudes = magnitudes[magnitudes["Station_Filter"]]
            if self.dist_filter:
                magnitudes = magnitudes[magnitudes["Dist_Filter"]]
            # Apply a custom version of the noise filter, in order to keep observations
            # where the signal would be expected to be above the noise threshold
            if self.noise_filter <= 0.0:
                raise AttributeError(
                    "Noise filter must be greater than 1 to use custom mag "
                    "r-squared filtering. Change 'only_used' to True, or "
                    f"set a noise filter (current = {self.noise_filter}"
                )
            for _, mag in magnitudes[~magnitudes["Noise_Filter"]].iterrows():
                # Correct noise amp for station correction
                noise_amp = (
                    mag["Noise_amp"]
                    * self.amp_multiplier
                    * np.power(10, mag["Station_Correction"])
                )
                # Calculate predicted amp
                att = self._get_attenuation(mag["Dist"])
                predicted_amp = np.power(10, (mean_mag - att))
                # If predicted amp is more than 5x larger than noise amp, keep
                # this observation for mag_r2 calculation
                if predicted_amp / noise_amp < 5:
                    magnitudes.drop(labels=mag.name)

        # Calculate amplitudes -- including station corrections!
        amps = (
            magnitudes[self.amp_feature].values
            * self.amp_multiplier
            * np.power(10, magnitudes["Station_Correction"])
        )

        dist = magnitudes["Dist"]
        att = self._get_attenuation(dist)

        # Find variance of log(amplitude) observations -- doing this in log
        # space to linearise the problem (so that r_squared is meaningful)
        log_amp_mean = np.log10(amps).mean()
        log_amp_variance = ((np.log10(amps) - log_amp_mean) ** 2).sum()

        # Calculate variance of log(amplitude) variations with respect to
        # amplitude vs. distance curve predicted by the calculated ML &
        # attenuation function
        mod_variance = ((np.log10(amps) - (mean_mag - att)) ** 2).sum()

        # Calculate the r-squared value (fraction of the log(amplitude)
        # variance that is explained by the predicted amplitude vs. distance
        # variation)
        r_squared = (log_amp_variance - mod_variance) / log_amp_variance

        return r_squared

# -*- coding: utf-8 -*-
"""
Module to produce a summary plot for local magnitude calculation from Wood-Anderson
corrected displacement amplitude measurements.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import matplotlib.pyplot as plt
import numpy as np


def amplitudes_summary(
    magnitudes, amp_feature, amp_multiplier, dist_err, r_squared, noise_measure="RMS"
):
    """
    Plot figure showing the measured signal amplitudes against distance from the event.

    Parameters
    ----------
    magnitudes : `pandas.DataFrame` object
        Contains P- and S-wave amplitude measurements for each component of each station
        in the station file, and local magnitude estimates calculated from them (output
        by calculate_magnitudes()). Note that the amplitude observations are raw, but
        the ML estimates derived from them include station corrections, if provided.
        Columns = ["epi_dist", "z_dist", "P_amp", "P_freq", "P_time",
                   "P_avg_amp", "P_filter_gain", "S_amp", "S_freq",
                   "S_time", "S_avg_amp", "S_filter_gain", "Noise_amp",
                   "is_picked", "ML", "ML_Err"], "Noise_Filter",
                   "Trace_Filter", "Station_Filter", "Dist_Filter", "Dist",
                   "Used"]
    amp_feature : {"S_amp", "P_amp"}
        Which phase amplitude measurement to use to calculate local magnitude.
        (Default "S_amp")
    amp_multiplier : float
        Factor by which to multiply all measured amplitudes.
    dist_err : float
        (Epicentral or hypocentral) distance uncertainty - calculated from the
        LocalGaussian location uncertainties.
    r_squared : float
        r-squared statistic describing the fit of the amplitude vs. distance curve
        predicted by the calculated mean_mag and chosen attenuation model to the
        measured amplitude observations. This is intended to be used to help
        discriminate between 'real' events, for which the predicted amplitude vs.
        distance curve should provide a good fit to the observations, from artefacts,
        which in general will not.
    noise_measure : {"RMS", "STD", "ENV"}, optional
        The method by which to measure the amplitude of the signal in the noise window:
        root-mean-square, standard deviation or average amplitude of the envelope of the
        signal. (Default "RMS")

    Returns
    -------
    fig : `matplotlib.pyplot.Figure` object
        Figure showing the measured amplitudes against distance from the event.
    ax : `matplotlib.axes.Axes` object
        Figure axes.

    """

    # Initiate figure
    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(111)

    # Correct noise amplitudes according to station corrections
    noise_amps = (
        magnitudes["Noise_amp"].values
        * amp_multiplier
        * np.power(10, magnitudes["Station_Correction"])
    )
    filter_gains = magnitudes[f"{amp_feature[0]}_filter_gain"]
    if not filter_gains.isnull().values.any():
        noise_amps /= filter_gains

    # Set to loglog scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Set axis tick label font size
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Plot noise amps
    noise_label = f"Noise amplitude ({noise_measure} amplitude in noise window)"
    ax.scatter(magnitudes["Dist"], noise_amps, marker="v", c="k", label=noise_label)

    # Plot median noise amplitude
    ax.axhline(
        np.median(noise_amps), linestyle=":", color="k", label="Median noise amplitude"
    )

    # Plot amplitude obs for used observations
    used_mags = magnitudes[magnitudes["Used"]]
    signal_label = (
        f"Signal amplitude (max amplitude in {amp_feature[0]}-wave signal window)"
    )
    # Plot amplitudes with station corrections applied
    _, _, bars = ax.errorbar(
        used_mags["Dist"],
        used_mags[amp_feature]
        * amp_multiplier
        * np.power(10, used_mags["Station_Correction"]),
        xerr=dist_err,
        yerr=noise_amps[magnitudes["Used"]],
        marker="x",
        label=signal_label,
    )
    _ = [errorbar.set_alpha(0.3) for errorbar in bars]

    # One label for each station, above highest observed amplitude; faff.
    ax, stns = label_stations(
        ax,
        used_mags.index,
        used_mags[amp_feature]
        * amp_multiplier
        * np.power(10, used_mags["Station_Correction"]),
        used_mags["Dist"],
    )

    # Plot amplitude obs for rejected observations (if there are any)
    rejected_mags = magnitudes[~magnitudes["Used"]]
    if len(rejected_mags) > 0:
        unused_label = f"Unused {amp_feature[0]}-wave amplitude observations"
        _, _, bars = ax.errorbar(
            rejected_mags["Dist"],
            rejected_mags[amp_feature]
            * amp_multiplier
            * np.power(10, rejected_mags["Station_Correction"]),
            xerr=dist_err,
            yerr=noise_amps[~magnitudes["Used"]],
            fmt="o",
            marker="x",
            c="gray",
            label=unused_label,
        )
        _ = [errorbar.set_alpha(0.3) for errorbar in bars]

        # Only label stations which were not already labelled
        rej_trids = []
        rej_amps = []
        rej_dists = []
        for i, tr_id in enumerate(rejected_mags.index):
            stn = tr_id[:-1]
            if stn in stns:
                continue
            else:
                rej_trids.append(tr_id)
                rej_amps.append(
                    rejected_mags[amp_feature].iloc[i]
                    * amp_multiplier
                    * np.power(10, rejected_mags["Station_Correction"].iloc[i])
                )
                rej_dists.append(rejected_mags["Dist"].iloc[i])

        # Only one label per new station; faff once again.
        ax, _ = label_stations(ax, rej_trids, rej_amps, rej_dists, rejected=True)

    # Label r-squared value
    ax.text(
        0.98,
        0.02,
        f"r-squared: {r_squared:.2f}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="w", alpha=0.8),
        va="bottom",
        ha="right",
        fontsize=16,
    )

    return fig, ax


def label_stations(ax, tr_ids, amps, dists, rejected=False):
    """
    Add station labels to the amplitudes vs. distance plot: only one label for each
    station, above the highest observed amplitude. Much faff.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes` object
        Axes on which to add the station labels.
    tr_ids : list of str
        List of trace ID's for which there are amplitude observations to label.
    amps : array-like (floats)
        Amplitudes (y-axis) coordinates
    dists : array-like (floats)
        Distances (x-axis) coordinates
    rejected : bool, optional
        Whether these are labels for rejected measurements (plotted in grey).

    Returns
    -------
    ax : `matplotlib.axes.Axes` object
        Now with the labels added
    stns : list
        List of stations that have been labelled.

    """

    stn = None
    stns = []
    comps = []
    for i, tr_id in enumerate(tr_ids):
        if not stn:
            stn = tr_id[:-1]
            stn_start = 0
            comps.append(tr_id[-1])
            continue
        elif tr_id[:-1] != stn:
            stn_end = i
            distance = dists[i - 1]
            amp = max(amps[stn_start:stn_end])
            compstring = ""
            for comp in comps:
                compstring += f"{comp},"
            label = f"{stn}[{compstring[:-1]}]"
            if not rejected:
                ax.annotate(
                    label, (distance, amp), ha="center", va="bottom", fontsize=8
                )
            else:
                ax.annotate(
                    label,
                    (distance, amp),
                    color="gray",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            stns.append(stn)
            stn = tr_id[:-1]
            stn_start = i
            comps = [tr_id[-1]]
            if i == len(tr_ids) - 1:
                distance = dists[i]
                amp = max(amps[stn_start:])
                compstring = ""
                for comp in comps:
                    compstring += f"{comp},"
                label = f"{stn}[{compstring[:-1]}]"
                if not rejected:
                    ax.annotate(
                        label, (distance, amp), ha="center", va="bottom", fontsize=8
                    )
                else:
                    ax.annotate(
                        label,
                        (distance, amp),
                        color="gray",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        elif i == len(tr_ids) - 1:
            stn = tr_id[:-1]
            comps.append(tr_id[-1])
            distance = dists[i]
            amp = max(amps[stn_start:])
            compstring = ""
            for comp in comps:
                compstring += f"{comp},"
            label = f"{stn}[{compstring[:-1]}]"
            if not rejected:
                ax.annotate(
                    label, (distance, amp), ha="center", va="bottom", fontsize=8
                )
            else:
                ax.annotate(
                    label,
                    (distance, amp),
                    color="gray",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            stns.append(stn)
        else:
            comps.append(tr_id[-1])

    return ax, stns

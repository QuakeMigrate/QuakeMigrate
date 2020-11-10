# -*- coding: utf-8 -*-
"""
Module to produce a summary plot for the phase picking.

:copyright:
    2020, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

import matplotlib.pyplot as plt
import numpy as np

import quakemigrate.util as util


def pick_summary(event, station, signal, picks, onsets, ttimes, windows):
    """
    Plot figure showing the filtered traces for each data component and the
    characteristic functions calculated from them (P and S) for each
    station. The search window to make a phase pick is displayed, along
    with the dynamic pick threshold (defined as a percentile of the
    background noise level), the phase pick time and its uncertainty (if
    made) and the Gaussian fit to the characteristic function.

    Parameters
    ----------
    event : str
        Unique identifier for the event.
    station : str
        Station code.
    signal : `numpy.ndarray` of int
        Seismic data for the Z N and E components.
    picks : pandas DataFrame object
        Phase pick times with columns ["Name", "Phase", "ModelledTime",
        "PickTime", "PickError", "SNR"]
        Each row contains the phase pick from one station/phase.
    onsets : `numpy.ndarray` of float
        Onset functions for each seismic phase, shape(nstations, nsamples).
    ttimes : list, [int, int]
        Modelled phase travel times.
    windows : dict of list, [int, int]
        Keys are phase. Indices specifying the window within which the pick was
        made.

    Returns
    -------
    fig : `matplotlib.Figure` object
        Figure showing basic phase picking information.

    """

    fig = plt.figure(figsize=(30, 15))

    # Create plot axes, ordering: [Z data, N data, E data, P onset, S onset]
    for i in [2, 1, 3, 4, 5]:
        ax = fig.add_subplot(3, 2, i+1)
    axes = fig.axes

    # Share P-pick x-axes and set title
    axes[0].get_shared_x_axes().join(axes[0], axes[3])
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].yaxis.set_ticks_position('none')
    axes[0].set_title("P phase", fontsize=22, fontweight="bold")

    # Share S-pick x-axes and set title
    for ax in axes[1:3]:
        ax.get_shared_x_axes().join(ax, axes[4])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.yaxis.set_ticks_position('none')
    axes[1].set_title("S phase", fontsize=22, fontweight="bold")

    # Add axis for text info
    text = fig.add_subplot(3, 2, 1)
    text.text(0.5, 0.8, f"Event: {event.uid}\nStation: {station}", ha="center",
              va="center", fontsize=22, fontweight="bold")

    # --- Grab event information once ---
    otime = event.otime
    times = signal[0].times(type="utcdatetime")
    mpl_times = signal[0].times(type="matplotlib")

    phases = [phase for phase, _ in onsets.items()]
    onsets = [onset for _, onset in onsets.items()]

    # --- Calculate plotting window ---
    # Estimate suitable windows based on ttimes
    min_t = otime + 0.5 * ttimes[0]
    max_t = otime + 1.5 * ttimes[-1]
    min_t_idx = np.argmin([abs(t - min_t) for t in times])
    max_t_idx = np.argmin([abs(t - max_t) for t in times])

    # Estimate suitable windows based on windows (10 sample pad is arbitrary)
    min_win_idx = np.min([v[0] for v in windows.values()]) - 10
    max_win_idx = np.max([v[-1] for v in windows.values()]) + 10
    # Take min and max
    min_idx = min(min_t_idx, min_win_idx)
    max_idx = max(max_t_idx, max_win_idx)
    # Ensure min and max are within length of trace
    if min_idx < 0:
        logging.debug(f"Min index is before start of trace for station "
                      f"{station}! {min_idx}")
        min_idx = 0
    if max_idx >= len(times):
        logging.debug(f"Max index is after end of trace for station "
                      f"{station}! {max_idx} / {len(times)}")
        max_idx = len(times) - 1

    # --- Plot waveforms ---
    times = [x.datetime for x in times]
    for i, (ax, comp) in enumerate(zip(axes[:3], ["Z", "[N,1]", "[E,2]"])):
        tr = signal.select(component=comp)
        if not bool(tr):
            continue
        y = tr[0].data
        ax.plot(times[min_idx:max_idx+1], y[min_idx:max_idx+1], c="k", lw=0.5,
                zorder=1)
        # Add label (SEED id)
        ax.text(0.015, 0.95, f"{tr[0].id}", transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc="w", alpha=0.8), va="top",
                ha="left", fontsize=18, zorder=2)
        # Set ylim
        y_min = min(y[min_win_idx:max_win_idx+1])
        y_max = max(y[min_win_idx:max_win_idx+1])
        ax.set_ylim(ymin=(y_min - 0.1 * abs(y_min)),
                    ymax=(y_max + 0.1 * abs(y_max)))

    # --- Plot onset functions ---
    # Handle case where only S phase is used
    n = 3
    if len(phases) == 1 and phases[0] == "S":
        n += 1

    # Loop through all relevant onset function axes
    for i, (ax, ph) in enumerate(zip(axes[n:5], phases)):
        # Plot onset functions
        y = onsets[i]
        ax.plot(times[min_idx:max_idx+1], y[min_idx:max_idx+1], c="k", lw=0.5,
                zorder=1)

        # Plot labels
        ax.text(0.015, 0.95, f"{ph} onset", transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc="w", alpha=0.8), va="top",
                ha="left", fontsize=18, zorder=2)

        # Plot threshold
        gau = event.picks["gaussfits"][station][ph]
        thresh = gau["PickThreshold"]
        ax.axhline(thresh, label="Pick threshold")
        text.text(0.05+i*0.5, 0.2, f"Threshold: {thresh:5.3f}", ha="left",
                  va="center", fontsize=18)

        # Plot gaussian fit to onset function
        if not gau["PickValue"] == -1:
            yy = util.gaussian_1d(gau["xdata"], gau["popt"][0],
                                  gau["popt"][1], gau["popt"][2])
            dt = [x.datetime for x in gau["xdata_dt"]]
            ax.plot(dt, yy)

        # Set ylim
        win = windows[ph]
        onset_max = max(onsets[i][win[0]:win[2]+1])
        y_max = max(onset_max, thresh)
        ax.set_ylim(0, y_max*1.1)

    # --- Plot predicted arrival times ---
    # Handle case where only a single phase is used
    ax_ind = range(5)
    colors = ["#F03B20", "#3182BD"]
    if len(phases) == 1:
        if phases[0] == "P":
            ax_ind = [0, 3]
            colors = [colors[0]]
        else:
            ax_ind = [1, 2, 4]
            colors = [colors[1]]

    # Loop through all relevant axes
    for ind in ax_ind:
        ax = axes[ind]
        # Plot model picks
        model_pick = otime + ttimes[0] if ind % 3 == 0 else otime + ttimes[-1]
        ph = phases[0] if ind % 3 == 0 else phases[-1]
        ax.axvline(model_pick.datetime, alpha=0.9, c="k",
                   label=f"Modelled {ph} arrival")
        # Plot event origin time if it is on plot:
        if mpl_times[min_idx] < otime.matplotlib_date:
            ax.axvline(event.otime.datetime, c="green",
                       label="Event origin time")
        # Plot pick windows
        win = windows[phases[0]] if ind % 3 == 0 else windows[phases[-1]]
        clr = colors[0] if ind % 3 == 0 else colors[-1]
        ax.axvspan(mpl_times[win[0]], mpl_times[win[2]], alpha=0.2, color=clr,
                   label="Picking window")
        # Set xlim
        ax.set_xlim(mpl_times[min_idx], mpl_times[max_idx])

    # --- Plot picks and summary information ---
    for i, pick in picks.iterrows():
        # Pick lines
        if pick["Phase"] == "P":
            c1, c2 = "#F03B20", "gray"
        else:
            c1, c2 = "gray", "#3182BD"
        if pick["PickTime"] != -1:
            for ind in ax_ind:
                ax = axes[ind]
                clr = c1 if ind % 3 == 0 else c2
                _plot_phase_pick(ax, pick, clr)
        # Calculate residual:
        if pick.PickTime == -1:
            resid = -1
        else:
            resid = pick.PickTime - pick.ModelledTime
        # Summary text
        text.text(0.1+i*0.5, 0.6, f"{pick.Phase} phase", ha="center",
                  va="center", fontsize=20, fontweight="bold")
        pick_info = (f"Pick time: {pick.PickTime}\n"
                     f"Pick error: {pick.PickError:5.3f} s\n"
                     f"Pick SNR: {pick.SNR:5.3f}\n"
                     f"Pick residual: {resid:5.3f} s")
        text.text(0.05+i*0.5, 0.4, pick_info, ha="left", va="center",
                  fontsize=18)
    text.set_axis_off()

    # Add legend
    for ind in ax_ind:
        if ind > 2:
            axes[ind].legend(fontsize=16, loc="upper right")

    fig.tight_layout(pad=1)
    plt.subplots_adjust(hspace=0)

    return fig


def _plot_phase_pick(ax, pick, clr):
    """
    Plot the phase pick time with the associated uncertainty.

    Parameters
    ----------
    ax : `matplotlib.Axes` object
        Axes on which to plot the pick.
    pick : `pandas.DataFrame` object
        Contains information on the phase pick.
    clr : str
        Colour to use when plotting the phase pick.

    """

    pick_time, pick_err = pick["PickTime"], pick["PickError"]

    ax.axvline((pick_time - pick_err/2).datetime, ls="--", c=clr)
    ax.axvline((pick_time + pick_err/2).datetime, ls="--", c=clr)
    ax.axvline((pick_time).datetime, c=clr, label=f"{pick.Phase} pick time")

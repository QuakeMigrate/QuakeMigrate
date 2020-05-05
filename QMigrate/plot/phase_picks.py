# -*- coding: utf-8 -*-
"""
Module to produce a summary plot for the phase picking.

"""

import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()


def pick_summary(event, station_uid, signal, picks, onsets, ttimes, window):
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
    station_uid : str
        Unique identifer for the station.
    signal : array of arrays
        Seismic data for the Z N and E components.
    picks : pandas DataFrame object
        Phase pick times with columns: ["Name", "Phase",
                                        "ModelledTime",
                                        "PickTime", "PickError",
                                        "SNR"]
        Each row contains the phase pick from one station/phase.
    onsets : array of arrays
        P- and S-phase onset functions for the event-station pair.
    ttimes : array, [int, int]
        Modelled phase travel times.
    window : array, [int, int]
        Indices specifying the window within which the pick was made.

    Returns
    -------
    fig : matplotlib Figure object
        Figure showing basic phase picking information.

    """

    fig = plt.figure(figsize=(30, 15))

    # Create plot axes, ordering: [Z data, N data, E data, P onset, S onset]
    for i in [2, 1, 3, 4, 5]:
        ax = fig.add_subplot(3, 2, i+1)
        ax.set_ylim([-1.1, 1.1])
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

    # --- Grab event information once ---
    otime = event.otime
    times = event.data.times(type="utcdatetime")

    # --- Plot data signal ---
    # Trim data to just around phase picks
    min_t = otime + 0.5 * ttimes[0]
    max_t = otime + 1.5 * ttimes[1]

    # Find indices for window in which to get normalising factor, then plot
    min_t_idx = np.argmin([abs(t - min_t) for t in times])
    max_t_idx = np.argmin([abs(t - max_t) for t in times])
    times = [x.datetime for x in times]
    for i, ax in enumerate(axes[:3]):
        # Funky maths to assign the signal data to correct axes
        y = signal[(i+2) % 3, :]
        # Get normalising factor within window
        norm = np.max(abs(y[min_t_idx:max_t_idx+1]))
        ax.plot(times, y / norm, c="k", lw=0.5, zorder=1)
    for i, (ax, ph) in enumerate(zip(axes[3:], ["P", "S"])):
        y = onsets[i]
        win = window[ph]
        # Get normalising factor within window
        norm = np.max(abs(y[win[0]:win[1]+1]))
        ax.plot(times, y / norm, c="k", lw=0.5, zorder=1)

    for ax in axes:
        ax.set_xlim([min_t.datetime, max_t.datetime])

    # --- Plot labels and fix limits ---
    shift = (max_t - min_t) * 0.01  # Fractional shift for text label
    for comp, ax in zip(["Z", "N", "E"], axes[:3]):
        ax.text((min_t+shift).datetime, 0.9, f"{station_uid}.BH{comp}",
                ha="left", va="center", zorder=2, fontsize=18)
        ax.set_ylim([-1.1, 1.1])
        ax.set_yticks(np.arange(-1, 1.5, 0.5))
    for ph, ax in zip(["P", "S"], axes[3:]):
        ax.text((min_t+shift).datetime, 1., f"{ph} onset", ha="left",
                va="center", zorder=2, fontsize=18)
        ax.set_ylim([-0.1, 1.1])
        ax.set_yticks(np.arange(0., 1.2, 0.2))

    # --- Plot predicted arrival times ---
    for i, ax in enumerate(axes):
        model_pick = otime + ttimes[0] if i % 3 == 0 else otime + ttimes[1]
        ax.axvline(model_pick.datetime, alpha=0.9, c="k",
                   label="Modelled pick time")

    # --- Plot picks and summary information ---
    text = fig.add_subplot(3, 2, 1)
    text.text(0.5, 0.8, f"Event: {event.uid}\nStation: {station_uid}",
              ha="center", va="center", fontsize=22, fontweight="bold")
    for i, pick in picks.iterrows():
        # Pick lines
        if pick["Phase"] == "P":
            c1, c2 = "#F03B20", "gray"
        else:
            c1, c2 = "gray", "#3182BD"
        if pick["PickTime"] != -1:
            for j, ax in enumerate(axes):
                clr = c1 if j % 3 == 0 else c2
                _plot_phase_pick(ax, pick, clr)

        # Summary text
        text.text(0.1+i*0.5, 0.6, f"{pick.Phase} phase", ha="center",
                  va="center", fontsize=20, fontweight="bold")
        pick_summary = (f"Pick time: {pick.PickTime}\n"
                        f"Pick error: {pick.PickError:5.3f} s\n"
                        f"Pick SNR: {pick.SNR:5.3f}")
        text.text(0.05+i*0.5, 0.45, pick_summary, ha="left", va="center",
                  fontsize=18)
    text.set_axis_off()

    for ax in axes[3:5]:
        ax.legend()
    fig.tight_layout(pad=4, w_pad=4)
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

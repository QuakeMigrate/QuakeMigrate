# -*- coding: utf-8 -*-
"""
Module to plot the triggered events on a decimated grid.

"""

import logging
import os

import matplotlib
try:
    os.environ["DISPLAY"]
    matplotlib.use("Qt5Agg")
except KeyError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from QMigrate.io import read_availability
import QMigrate.util as util


def trigger_summary(events, starttime, endtime, run, marginal_window,
                    minimum_repeat, detection_threshold, normalise_coalescence,
                    lut, data, region, savefig):
    """
    Plots the data from a .scanmseed file with annotations illustrating the
    trigger results: event triggers and marginal windows on the coalescence
    traces, and map and cross section view of the gridded triggered earthquake
    locations.

    Parameters
    ----------
    events : `pandas.DataFrame`
        Triggered events information.
        Columns: ["EventNum", "CoaTime", "COA_V", "COA_X", "COA_Y", "COA_Z",
                  "MinTime", "MaxTime", "COA", "COA_NORM", "EventID"].
    starttime : `obspy.UTCDateTime`
        Start time of trigger run.
    endtime : `obspy.UTCDateTime`
        End time of trigger run.
    run : `QMigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    marginal_window : float
        Estimate of time error over which to marginalise the coalescence.
    minimum_repeat : float, optional
        Minimum time interval between triggered events.
    detection_threshold : float
        Coalescence value above which to trigger events.
    normalise_coalescence : bool
        If True, use coalescence normalised by the average background noise.
    lut : `QMigrate.lut.LUT` object
        Contains the traveltime lookup tables for P- and S-phases, computed for
        some pre-defined velocity model.
    data : `pandas.DataFrame`
        Data output by detect() -- decimated scan.
        Columns: ["COA", "COA_N", "X", "Y", "Z"]
    region : array
        Geographical region within which earthquakes have been triggered.
    savefig : bool
        Output the plot as a file. The plot is shown by default, and not saved.

    """

    dt = pd.to_datetime(data["DT"].astype(str))

    fig = plt.figure(figsize=(30, 15))

    # Create plot axes, ordering: [COA, COA_N, AVAIL, XY, XZ, YZ]
    for row in [0, 3, 6]:
        spec = GridSpec(9, 18).new_subplotspec((row, 9), colspan=9, rowspan=3)
        ax = fig.add_subplot(spec)
        ax.set_xlim([starttime.datetime, endtime.datetime])
    lut.plot(fig, (9, 18))
    axes = fig.axes

    # Share coalescence and availability axes
    for ax in axes[0:2]:
        ax.get_shared_x_axes().join(ax, axes[2])

    lab = "Maximum coalescence"
    axes[0].plot(dt, data.COA, c="k", lw=0.01, label=lab, alpha=0.8,
                 zorder=10)
    axes[1].plot(dt, data.COA_N, c="k", lw=0.01, alpha=0.8,
                 zorder=10)

    try:
        logging.info("\tReading in .StationAvailability...")
        availability = read_availability(run, starttime, endtime)
        available = availability.sum(axis=1).astype(int)
        times = list(pd.to_datetime(available.index))

        # Handle last step
        available = available.values
        available = np.append(available, [available[-1]])
        times.append(endtime.datetime)

        axes[2].step(times, available, c="green", where="post")
        axes[2].set_ylim([0, max(available) + 1])
    except util.NoStationAvailabilityDataException as e:
        logging.info(e.msg)

    if events is not None:
        for i, event in events.iterrows():
            lab1 = "Minimum repeat window" if i == 0 else ""
            lab2 = "Marginal window" if i == 0 else ""
            lab3 = "Triggered event" if i == 0 else ""

            min_dt = event["MinTime"].datetime
            max_dt = event["MaxTime"].datetime
            mw_stt = (event["CoaTime"] - marginal_window).datetime
            mw_end = (event["CoaTime"] + marginal_window).datetime
            for ax in axes[:2]:
                ax.axvspan(min_dt, mw_stt, label=lab1, alpha=0.2, color="#F03B20")
                ax.axvspan(mw_end, max_dt, alpha=0.2, color="#F03B20")
                ax.axvspan(mw_stt, mw_end, label=lab2, alpha=0.2, color="#3182BD")
                ax.axvline(event["CoaTime"].datetime, label=lab3, lw=0.01,
                           alpha=0.4)

    axes[0].text(0.01, 0.925, "Maximum coalescence", ha="left", va="center",
                 transform=axes[0].transAxes, fontsize=18)
    axes[0].set_ylabel("Coalescence", fontsize=14)
    axes[1].text(0.01, 0.925, "Normalised maximum coalescence", ha="left",
                 va="center", transform=axes[1].transAxes, fontsize=18)
    axes[1].set_ylabel("Normalised coalescence", fontsize=14)
    axes[2].text(0.01, 0.925, "Station availability", ha="left", va="center",
                 transform=axes[2].transAxes, fontsize=18)
    axes[2].set_ylabel("Available stations", fontsize=14)
    axes[2].set_xlabel("DateTime", fontsize=14)

    if events is not None:
        ax_i = 1 if normalise_coalescence else 0
        axes[ax_i].step(dt, detection_threshold, where="mid", c="g",
                        label="Detection threshold")

        # Get bounds for cmap
        vmin, vmax = events["COA_V"].min(), events["COA_V"].max()

        # Plotting the scatter of the earthquake locations
        x, y, z = events["COA_X"], events["COA_Y"], events["COA_Z"]
        c = events["COA_V"]
        sc = axes[3].scatter(x, y, s=50, c=c, cmap="viridis", vmin=vmin,
                             vmax=vmax)
        axes[4].scatter(x, z, s=50, c=c, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[5].scatter(z, y, s=50, c=c, cmap="viridis", vmin=vmin, vmax=vmax)

        # --- Add colourbar ---
        spec = GridSpec(9, 18).new_subplotspec((2, 7), colspan=1, rowspan=5)
        cax = fig.add_subplot(spec)
        cax.set_axis_off()
        cbar = fig.colorbar(sc, ax=cax, orientation="vertical", fraction=0.4)
        cbar.ax.set_ylabel("Peak coalescence value", rotation=90, fontsize=14)

    # --- Write summary information ---
    text = plt.subplot2grid((9, 18), (0, 0), colspan=8, rowspan=2, fig=fig)
    if len(set(detection_threshold)) == 1:
        threshold = f"{detection_threshold[0]} (static)"
    else:
        threshold = "dynamic"
    text.text(0.5, 0.75, f"Trigger threshold: {threshold}", ha="center",
              va="center", fontsize=22)
    text.text(0.5, 0.6, f"Marginal window: {marginal_window} s", ha="center",
              va="center", fontsize=22)
    text.text(0.5, 0.45, f"Minimum repeat time: {minimum_repeat} s",
              ha="center", va="center", fontsize=22)
    trig = "normalised coalescence" if normalise_coalescence else "coalescence"
    count = len(events) if events is not None else 0
    text.text(0.5, 0.2, (f"Triggered {count} event(s) on the {trig}"
                         " trace."), ha="center", va="center", fontsize=22)
    text.set_axis_off()

    axes[0].legend(loc=1, fontsize=14)
    fig.tight_layout(pad=1, h_pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Save figure or open interactive matplotlib window
    if savefig:
        fpath = run.path / "trigger" / run.subname / "summaries"
        fpath.mkdir(exist_ok=True, parents=True)
        fstem = f"{run.name}_Trigger"
        file = (fpath / fstem).with_suffix(".pdf")
        plt.savefig(str(file))
    else:
        plt.show()

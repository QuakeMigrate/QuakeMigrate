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
from pandas.plotting import register_matplotlib_converters


from QMigrate.io import read_availability
import QMigrate.util as util


register_matplotlib_converters()


def trigger_summary(events, starttime, endtime, run, marginal_window,
                    min_event_interval, detection_threshold,
                    normalise_coalescence, lut, data, region, savefig):
    """
    Plots the data from a .scanmseed file with annotations illustrating the
    trigger results: event triggers and marginal windows on the coalescence
    traces, and map and cross section view of the gridded triggered earthquake
    locations.

    Parameters
    ----------
    events : `pandas.DataFrame`
        Triggered events information, columns: ["EventID", "CoaTime",
        "TRIG_COA", "COA_X", "COA_Y", "COA_Z", "MinTime", "MaxTime", "COA",
        "COA_NORM"].
    starttime : `obspy.UTCDateTime`
        Start time of trigger run.
    endtime : `obspy.UTCDateTime`
        End time of trigger run.
    run : `QMigrate.io.Run` object
        Light class encapsulating i/o path information for a given run.
    marginal_window : float
        Estimate of time error over which to marginalise the coalescence.
    min_event_interval : float
        Minimum time interval between triggered events.
    detection_threshold : array-like
        Coalescence value above which to trigger events.
    normalise_coalescence : bool
        If True, use coalescence normalised by the average background noise.
    lut : `QMigrate.lut.LUT` object
        Contains the traveltime lookup tables for P- and S-phases, computed for
        some pre-defined velocity model.
    data : `pandas.DataFrame`
        Data output by detect() -- decimated scan, columns ["COA", "COA_N",
        "X", "Y", "Z"]
    region : list
        Geographical region within which earthquakes have been triggered.
    savefig : bool
        Output the plot as a file. The plot is shown by default, and not saved.

    """

    dt = pd.to_datetime(data["DT"].astype(str))

    fig = plt.figure(figsize=(30, 15))
    gs = (9, 18)

    # Create plot axes, ordering: [COA, COA_N, AVAIL, XY, XZ, YZ]
    for row in [0, 3, 6]:
        ax = plt.subplot2grid(gs, (row, 8), colspan=10, rowspan=3, fig=fig)
        ax.set_xlim([starttime.datetime, endtime.datetime])

    # --- Plot LUT, coalescence traces, and station availability ---
    lut.plot(fig, gs)
    axes = fig.axes
    for ax in axes[:2]:
        ax.get_shared_x_axes().join(ax, axes[2])
    # for ax, data, label in zip(fig.axes[:2], )
    _plot_coalescence(axes[0], dt, data.COA, "Maximum coalescence")
    _plot_coalescence(axes[1], dt, data.COA_N,
                      "Normalised maximum coalescence")
    try:
        logging.info("\n\t    Reading in .StationAvailability...")
        availability = read_availability(run, starttime, endtime)
        _plot_station_availability(axes[2], availability, endtime)
    except util.NoStationAvailabilityDataException as e:
        logging.info(e)

    # --- Plot event scatter on LUT and windows on coalescence traces ---
    if events is not None:
        _plot_event_windows(axes[:2], events, marginal_window)
        _plot_event_scatter(fig, events)

        # Add trigger threshold to the correct coalescence trace
        ax_i = 1 if normalise_coalescence else 0
        axes[ax_i].step(dt, detection_threshold, where="mid", c="g",
                        label="Detection threshold")

    # --- Write summary information ---
    text = plt.subplot2grid(gs, (0, 0), colspan=8, rowspan=2, fig=fig)
    st, et = [t.strftime("%Y-%m-%d %H:%M:%S") for t in (starttime, endtime)]
    text.text(0.42, 0.8, f"{st}  -  {et}", fontsize=20, fontweight="bold",
              ha="center")
    _plot_text_summary(text, events, detection_threshold, marginal_window,
                       min_event_interval, normalise_coalescence)

    fig.axes[0].legend(loc=1, fontsize=14)
    fig.tight_layout(pad=1, h_pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # --- Adjust cross sections to match map aspect ratio ---
    # Get left, bottom, width, height of each subplot bounding box
    xy_left, xy_bottom, xy_width, xy_height = fig.axes[3].get_position().bounds
    xz_l, xz_b, xz_w, xz_h = fig.axes[4].get_position().bounds
    yz_l, yz_b, _, _ = fig.axes[5].get_position().bounds
    # Find height and width spacing of subplots in figure coordinates
    hdiff = yz_b - (xz_b + xz_h)
    wdiff = yz_l - (xz_l + xz_w)
    # Adjust bottom of xz cross section (if bottom of map has moved up)
    new_xz_bottom = xy_bottom - hdiff - xz_h
    fig.axes[4].set_position([xy_left, new_xz_bottom, xy_width, xz_h])
    # Adjust left of yz cross section (if right side of map has moved left)
    new_yz_left = xy_left + xy_width + wdiff
    # Take this opportunity to ensure the height of both cross sections is
    # equal by adjusting yz width (almost there from gridspec maths already)
    new_yz_width = xz_h * (fig.get_size_inches()[1]
                           / fig.get_size_inches()[0])
    fig.axes[5].set_position([new_yz_left, xy_bottom, new_yz_width, xy_height])

    # Save figure or open interactive matplotlib window
    if savefig:
        fpath = run.path / "trigger" / run.subname / "summaries"
        fpath.mkdir(exist_ok=True, parents=True)
        fstem = f"{run.name}_{starttime.year}_{starttime.julday:03d}_Trigger"
        file = (fpath / fstem).with_suffix(".pdf")
        plt.savefig(str(file))
    else:
        plt.show()


def _plot_station_availability(ax, availability, endtime):
    """
    Utility function to handle all aspects of plotting the station
    availability.

    Parameters
    ----------
    ax : `~matplotlib.Axes` object
        Axes on which to plot the waveform gather.
    availability : `~pandas.DataFrame` object
        Dataframe containing the availability of stations through time.
    endtime : `~obspy.UTCDateTime`
        End time of trigger run.

    """

    available = availability.sum(axis=1).astype(int)
    times = list(pd.to_datetime(available.index))

    # Handle last step
    available = available.values
    available = np.append(available, [available[-1]])
    times.append(endtime.datetime)
    ax.step(times, available, c="green", where="post")

    _add_plot_tag(ax, "Station availability")
    ax.set_ylim([int(min(available)*0.8), int(max(available)*1.1)])
    ax.set_yticks(range(int(min(available)*0.8), int(max(available)*1.1)+1))
    ax.xaxis.set_major_formatter(util.DateFormatter("%H:%M:%S.{ms}", 2))
    ax.set_xlabel("DateTime", fontsize=14)
    ax.set_ylabel("Available stations", fontsize=14)


def _plot_coalescence(ax, dt, data, label):
    """
    Utility function to bring plotting of coalescence into one place.

    Parameters
    ----------
    ax : `~matplotlib.Axes` object
        Axes on which to plot the coalescence traces.
    dt : array-like
        Timestamps of the coalescence data.
    data : array-like
        Coalescence data to plot.
    label : str
        y-axis label.

    """

    if label == "Maximum coalescence":
        ax.plot(dt, data, c="k", lw=0.01, label=label, alpha=0.8, zorder=10)
    else:
        ax.plot(dt, data, c="k", lw=0.01, alpha=0.8, zorder=10)
    _add_plot_tag(ax, label)
    ax.set_ylabel(label, fontsize=14)
    ax.xaxis.set_major_formatter(util.DateFormatter("%H:%M:%S.{ms}", 2))


def _add_plot_tag(ax, tag):
    """
    Utility function to plot tags on data traces.

    Parameters
    ----------
    ax : `~matplotlib.Axes` object
        Axes on which to plot the tag.
    tag : str
        Text to go in the tag.

    """

    ax.text(0.01, 0.925, tag, ha="left", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="w", alpha=0.8), fontsize=18)


def _plot_event_scatter(fig, events):
    """
    Utility function for plotting the triggered events as a scatter on the
    LUT cross-sections.

    Parameters
    ----------
    fig : `~matplotlib.Figure` object
        Figure containing axes on which to plot event scatter.
    events : `~pandas.DataFrame` object
        Dataframe of triggered events.

    """

    # Get bounds for cmap
    vmin, vmax = events["TRIG_COA"].min(), events["TRIG_COA"].max()

    # Plotting the scatter of the earthquake locations
    x, y, z = events["COA_X"], events["COA_Y"], events["COA_Z"]
    c = events["TRIG_COA"]
    sc = fig.axes[3].scatter(x, y, s=50, c=c, vmin=vmin, vmax=vmax)
    fig.axes[4].scatter(x, z, s=50, c=c, vmin=vmin, vmax=vmax)
    fig.axes[5].scatter(z, y, s=50, c=c, vmin=vmin, vmax=vmax)

    # --- Add colourbar ---
    cax = plt.subplot2grid((9, 18), (7, 5), colspan=2, rowspan=2, fig=fig)
    cax.set_axis_off()
    cb = fig.colorbar(sc, ax=cax, orientation="horizontal", fraction=0.8,
                      aspect=8)
    cb.ax.set_xlabel("Peak coalescence value", rotation=0, fontsize=14)


def _plot_event_windows(axes, events, marginal_window):
    """
    Utility function for plotting the marginal event window and minimum event
    interval for triggered events.

    Parameters
    ----------
    axes : list of `~matplotlib.Axes` objects
        Axes on which to plot the event windows.
    events : `~pandas.DataFrame` object
        Dataframe of triggered events.
    marginal_window : float
        Estimate of time error over which to marginalise the coalescence.

    """

    for i, event in events.iterrows():
        lab1 = "Minimum event interval" if i == 0 else ""
        lab2 = "Marginal window" if i == 0 else ""
        lab3 = "Triggered event" if i == 0 else ""

        min_dt = event["MinTime"].datetime
        max_dt = event["MaxTime"].datetime
        mw_stt = (event["CoaTime"] - marginal_window).datetime
        mw_end = (event["CoaTime"] + marginal_window).datetime
        for ax in axes:
            ax.axvspan(min_dt, mw_stt, label=lab1, alpha=0.2, color="#F03B20")
            ax.axvspan(mw_end, max_dt, alpha=0.2, color="#F03B20")
            ax.axvspan(mw_stt, mw_end, label=lab2, alpha=0.2, color="#3182BD")
            ax.axvline(event["CoaTime"].datetime, label=lab3, lw=0.01,
                       alpha=0.4)


def _plot_text_summary(ax, events, threshold, marginal_window,
                       min_event_interval, normalise_coalescence):
    """
    Utility function to plot the event summary information.

    Parameters
    ----------
    ax : `~matplotlib.Axes` object
        Axes on which to plot the text summary.
    events : `~pandas.DataFrame` object
        DataFrame of triggered events.
    threshold : array-like
        Coalescence value above which to trigger events.
    marginal_window : float
        Estimate of time error over which to marginalise the coalescence.
    min_event_interval : float
        Minimum time interval between triggered events.
    normalise_coalescence : bool
        If True, use coalescence normalised by the average background noise.

    """

    # Get threshold info
    if len(set(threshold)) == 1:
        threshold = f"{threshold[0]} (static)"
    else:
        threshold = "dynamic"

    # Get trigger on and event count info
    trig = "normalised coalescence" if normalise_coalescence else "coalescence"
    count = len(events) if events is not None else 0

    with plt.rc_context({"font.size": 18}):
        ax.text(0.45, 0.65, "Trigger threshold:", ha="right", va="center")
        ax.text(0.47, 0.65, f"{threshold}", ha="left", va="center")
        ax.text(0.45, 0.5, "Marginal window:", ha="right", va="center")
        ax.text(0.47, 0.5, f"{marginal_window} s", ha="left", va="center")
        ax.text(0.45, 0.35, "Minimum event interval:", ha="right", va="center")
        ax.text(0.47, 0.35, f"{min_event_interval} s", ha="left", va="center")
        ax.text(0.42, 0.15, f"Triggered {count} event(s) on the {trig} trace.",
                ha="center", va="center")
    ax.set_axis_off()

# -*- coding: utf-8 -*-
"""
Module to plot the triggered events on a decimated grid.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from quakemigrate.io import read_availability
import quakemigrate.util as util


@util.timeit("info")
def trigger_summary(
    events,
    starttime,
    endtime,
    run,
    marginal_window,
    min_event_interval,
    detection_threshold,
    normalise_coalescence,
    lut,
    data,
    region,
    discarded_events,
    interactive,
    xy_files=None,
    plot_all_stns=True,
):
    """
    Plots the data from a .scanmseed file with annotations illustrating the trigger
    results: event triggers and marginal windows on the coalescence traces, and map and
    cross section view of the gridded triggered earthquake locations.

    Parameters
    ----------
    events : `pandas.DataFrame`
        Triggered events information, columns: ["EventID", "CoaTime", "TRIG_COA",
        "COA_X", "COA_Y", "COA_Z", "MinTime", "MaxTime", "COA", "COA_NORM"].
    starttime : `obspy.UTCDateTime`
        Start time of trigger run.
    endtime : `obspy.UTCDateTime`
        End time of trigger run.
    run : :class:`~quakemigrate.io.core.Run` object
        Light class encapsulating i/o path information for a given run.
    marginal_window : float
        Time window over which to marginalise the 4D coalescence function.
    min_event_interval : float
        Minimum time interval between triggered events.
    detection_threshold : array-like
        Coalescence value above which to trigger events.
    normalise_coalescence : bool
        If True, use coalescence normalised by the average coalescence value in the 3-D
        grid at each timestep.
    lut : :class:`~quakemigrate.lut.lut.LUT` object
        Contains the traveltime lookup tables for the selected seismic phases, computed
        for some pre-defined velocity model.
    data : `pandas.DataFrame`
        Data output by :func:`~quakemigrate.signal.scan.QuakeScan.detect()` --
        continuous scan, columns: ["COA", "COA_N", "X", "Y", "Z"]
    region : list
        Geographical region within which to trigger earthquakes; events located outside
        this region will be discarded.
    discarded_events : `pandas.DataFrame`
        Discarded triggered events information, columns: ["EventID", "CoaTime",
        "TRIG_COA", "COA_X", "COA_Y", "COA_Z", "MinTime", "MaxTime", "COA", "COA_NORM"].
    interactive : bool
        Toggles whether to produce an interactive plot.
    xy_files : str, optional
        Path to comma-separated value file (.csv) containing a series of coordinate
        files to plot. Columns: ["File", "Color", "Linewidth", "Linestyle"], where
        "File" is the absolute path to the file containing the coordinates to be
        plotted. E.g: "/home/user/volcano_outlines.csv,black,0.5,-". Each .csv
        coordinate file should contain coordinates only, with columns: ["Longitude",
        "Latitude"]. E.g.: "-17.5,64.8". Lines pre-pended with ``#`` will be treated as
        a comment - this can be used to include references. See the
        Volcanotectonic_Iceland example XY_files for a template.\n
        .. note:: Do not include a header line in either file.
    plot_all_stns : bool, optional
        If true, plot all stations used for detect. Otherwise, only plot stations which
        for which some data was available during the trigger time window. NOTE: if no
        station availability data is found, all stations in the LUT will be plotted.
        (Default, True)

    """

    dt = pd.to_datetime(data["DT"].astype(str)).values

    fig = plt.figure(figsize=(30, 15))
    gs = (9, 18)

    logging.debug(discarded_events)

    # Create plot axes, ordering: [COA, COA_N, AVAIL, XY, XZ, YZ]
    for row in [0, 3, 6]:
        ax = plt.subplot2grid(gs, (row, 8), colspan=10, rowspan=3, fig=fig)
        ax.set_xlim([starttime.datetime, endtime.datetime])

    # --- Plot LUT, coalescence traces, and station availability ---
    for ax in fig.axes[:2]:
        ax.sharex(fig.axes[2])
    _plot_coalescence(fig.axes[0], dt, data.COA.values, "Maximum coalescence")
    _plot_coalescence(
        fig.axes[1], dt, data.COA_N.values, "Normalised maximum coalescence"
    )
    try:
        availability = read_availability(run, starttime, endtime)
        _plot_station_availability(fig.axes[2], availability, endtime)
    except util.NoStationAvailabilityDataException as e:
        logging.info(e)
        availability = None

    # Use station availability to work out which stations to plot
    if availability is not None:
        station_list = []
        if not plot_all_stns:
            for col, ava in availability.iteritems():
                if np.any(ava == 1):
                    station_list.append(col.split("_")[0])
        else:
            station_list = [col.split("_")[0] for col in availability.columns]
        station_list = list(set(sorted(station_list)))
        lut.plot(fig, gs, station_list=station_list)
    else:
        lut.plot(fig, gs)

    # --- Plot xy files on map ---
    _plot_xy_files(xy_files, fig.axes[3])

    # --- Plot trigger region (if any) ---
    if region is not None:
        _plot_trigger_region(fig.axes[3:], region)
        _plot_event_windows(
            fig.axes[:2], discarded_events, marginal_window, discarded=True
        )
        _plot_event_scatter(fig, discarded_events, discarded=True)

    # --- Plot event scatter on LUT and windows on coalescence traces ---
    if events is not None:
        _plot_event_windows(fig.axes[:2], events, marginal_window)
        _plot_event_scatter(fig, events)

        # Add trigger threshold to the correct coalescence trace
        ax_i = 1 if normalise_coalescence else 0
        fig.axes[ax_i].step(
            dt, detection_threshold, where="mid", c="g", label="Detection threshold"
        )

    # --- Write summary information ---
    text = plt.subplot2grid(gs, (0, 0), colspan=8, rowspan=2, fig=fig)
    st, et = [t.strftime("%Y-%m-%d %H:%M:%S") for t in (starttime, endtime)]
    text.text(0.42, 0.8, f"{st}  -  {et}", fontsize=20, fontweight="bold", ha="center")
    _plot_text_summary(
        text,
        events,
        detection_threshold,
        marginal_window,
        min_event_interval,
        normalise_coalescence,
    )

    # --- Handle legend for coalescence trace plot ---
    handles, labels = fig.axes[ax_i].get_legend_handles_labels()
    uniq_labels = dict(zip(labels, handles))
    fig.axes[ax_i].legend(
        uniq_labels.values(), uniq_labels.keys(), loc=1, fontsize=14, framealpha=0.85
    ).set_zorder(20)

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
    new_yz_width = xz_h * (fig.get_size_inches()[1] / fig.get_size_inches()[0])
    fig.axes[5].set_position([new_yz_left, xy_bottom, new_yz_width, xy_height])

    # Save figure
    fpath = run.path / "trigger" / run.subname / "summaries"
    fpath.mkdir(exist_ok=True, parents=True)
    fstem = f"{run.name}_{starttime.year}_{starttime.julday:03d}_Trigger"
    file = (fpath / fstem).with_suffix(".pdf")
    plt.savefig(str(file))

    if interactive:
        plt.show()

    plt.close(fig)


def _plot_station_availability(ax, availability, endtime):
    """
    Utility function to handle all aspects of plotting the station availability.

    Parameters
    ----------
    ax : `matplotlib.Axes` object
        Axes on which to plot the waveform gather.
    availability : `pandas.DataFrame` object
        Dataframe containing the availability of stations through time.
    endtime : `obspy.UTCDateTime`
        End time of trigger run.

    """

    # Get list of phases from station availability dataframe
    phases = sorted(set([col_name.split("_")[1] for col_name in availability.columns]))
    logging.debug(f"\t\t    Found phases: {phases}")

    # Sort out plotting options based on the number of phases
    if len(phases) > 2:
        logging.warning(
            "\t\t    Only P and/or S are currently supported! "
            "Plotting by station only."
        )
        phases = ["*"]
        colours = ["green"]
        divideby = len(phases)
    elif len(phases) == 1:
        if phases[0] == "P":
            colours = ["#F03B20"]
        else:
            colours = ["#3182BD"]
    elif (
        availability.filter(like=f"_{phases[0]}").values
        == availability.filter(like=f"_{phases[1]}").values
    ).all():
        logging.info(
            "\t\t    Station availability is identical for both "
            "phases; plotting by station only."
        )
        divideby = len(phases)
        phases = ["*"]
        colours = ["green"]
    else:
        colours = ["#F03B20", "#3182BD"]

    # Loop through phases and plot
    max_ava = []
    min_ava = []
    for phase, colour in zip(phases, colours):
        ph_availability = availability.filter(regex=f"_{phase}$")

        available = ph_availability.sum(axis=1).astype(int)
        times = list(pd.to_datetime(available.index).tz_localize(None))

        # If plotting by station, divide by # of phases
        if phases[0] == "*":
            # This can lead to incorrect value (e.g. if 2 / 3 phases are
            # available for a station). But not important enough to faff with.
            available = (available / divideby).astype(int)

        # Handle last step
        available = available.values
        available = np.append(available, [available[-1]])
        times.append(pd.to_datetime(endtime.datetime))
        logging.debug(times)
        ax.step(times, available, c=colour, where="post", label=phase)

        max_ava.append(max(available))
        min_ava.append(min(available))

    # Plot formatting
    _add_plot_tag(ax, "Station availability")
    ax.set_ylim([int(min(min_ava) * 0.8), int(np.ceil(max(max_ava) * 1.1))])
    ax.set_yticks(range(int(min(min_ava) * 0.8), int(np.ceil(max(max_ava) * 1.1)) + 1))
    ax.xaxis.set_major_formatter(util.DateFormatter("%H:%M:%S.{ms}", 2))
    ax.set_xlabel("DateTime", fontsize=14)
    ax.set_ylabel("Available stations", fontsize=14)
    if phases[0] != "*":
        ax.legend(loc=1, fontsize=14, framealpha=0.85).set_zorder(20)


def _plot_coalescence(ax, dt, data, label):
    """
    Utility function to bring plotting of coalescence trace into one place.

    Parameters
    ----------
    ax : `matplotlib.Axes` object
        Axes on which to plot the coalescence traces.
    dt : array-like
        Timestamps of the coalescence data.
    data : array-like
        Coalescence data to plot.
    label : str
        y-axis label.

    """

    ax.plot(dt, data, c="k", lw=0.01, label="Coalesence value", alpha=0.8, zorder=10)
    _add_plot_tag(ax, label)
    ax.set_ylabel(label, fontsize=14)
    ax.xaxis.set_major_formatter(util.DateFormatter("%H:%M:%S.{ms}", 2))


def _add_plot_tag(ax, tag):
    """
    Utility function to plot tags on data traces.

    Parameters
    ----------
    ax : `matplotlib.Axes` object
        Axes on which to plot the tag.
    tag : str
        Text to go in the tag.

    """

    ax.text(
        0.01,
        0.925,
        tag,
        ha="left",
        va="center",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="w", alpha=0.8),
        fontsize=18,
        zorder=20,
    )


def _plot_event_scatter(fig, events, discarded=False):
    """
    Utility function for plotting the triggered events as a scatter on the LUT map and
    cross-sections.

    Parameters
    ----------
    fig : `matplotlib.Figure` object
        Figure containing axes on which to plot event scatter.
    events : `pandas.DataFrame` object
        Dataframe of triggered events.
    discarded : bool, optional
        Whether supplied events are discarded (due to being outside the trigger region,
        or outside the trigger time window).

    """

    if discarded:
        x, y, z = events[["COA_X", "COA_Y", "COA_Z"]].values.T
        # Plot on XY
        fig.axes[3].scatter(x, y, s=50, c="grey")
        # Plot on XZ
        fig.axes[4].scatter(x, z, s=50, c="grey")
        # Plot on YZ
        fig.axes[5].scatter(z, y, s=50, c="grey")

    else:
        # Get bounds for cmap - hack to prevent inconsistent color being
        # assigned when only a single event has been triggered.
        vmin, vmax = (
            events["TRIG_COA"].min() * 0.999,
            events["TRIG_COA"].max() * 1.001,
        )

        # Plotting the scatter of the earthquake locations
        x, y, z = events[["COA_X", "COA_Y", "COA_Z"]].values.T
        c = events["TRIG_COA"].values
        sc = fig.axes[3].scatter(x, y, s=50, c=c, vmin=vmin, vmax=vmax)
        fig.axes[4].scatter(x, z, s=50, c=c, vmin=vmin, vmax=vmax)
        fig.axes[5].scatter(z, y, s=50, c=c, vmin=vmin, vmax=vmax)

        # --- Add colourbar ---
        cax = plt.subplot2grid((9, 18), (7, 5), colspan=2, rowspan=2, fig=fig)
        cax.set_axis_off()
        cb = fig.colorbar(sc, ax=cax, orientation="horizontal", fraction=0.8, aspect=8)
        cb.ax.set_xlabel("Peak coalescence value", rotation=0, fontsize=14)


def _plot_event_windows(axes, events, marginal_window, discarded=False):
    """
    Utility function for plotting the marginal event window and minimum event interval
    for triggered events.

    Parameters
    ----------
    axes : list of `matplotlib.Axes` objects
        Axes on which to plot the event windows.
    events : `pandas.DataFrame` object
        Dataframe of triggered events.
    marginal_window : float
        Estimate of time error over which to marginalise the coalescence.
    discarded : bool, optional
        Whether supplied events are discarded (due to being outside the trigger region,
        or outside the trigger time window).

    """

    for _, event in events.iterrows():
        min_dt = event["MinTime"].datetime
        max_dt = event["MaxTime"].datetime
        mw_stt = (event["CoaTime"] - marginal_window).datetime
        mw_end = (event["CoaTime"] + marginal_window).datetime
        for ax in axes:
            if discarded:
                ax.axvspan(min_dt, max_dt, alpha=0.2, color="grey")
                ax.axvline(event["CoaTime"].datetime, lw=0.01, alpha=0.4, color="grey")
            else:
                ax.axvspan(
                    min_dt,
                    mw_stt,
                    label="Minimum event interval",
                    alpha=0.2,
                    color="#F03B20",
                )
                ax.axvspan(mw_end, max_dt, alpha=0.2, color="#F03B20")
                ax.axvspan(
                    mw_stt, mw_end, label="Marginal window", alpha=0.2, color="#3182BD"
                )
                ax.axvline(
                    event["CoaTime"].datetime,
                    label="Triggered event",
                    lw=0.01,
                    alpha=0.4,
                    color="#1F77B4",
                )


def _plot_text_summary(
    ax, events, threshold, marginal_window, min_event_interval, normalise_coalescence
):
    """
    Utility function to plot the trigger summary information.

    Parameters
    ----------
    ax : `matplotlib.Axes` object
        Axes on which to plot the text summary.
    events : `pandas.DataFrame` object
        DataFrame of triggered events.
    threshold : array-like
        Coalescence value above which to trigger events.
    marginal_window : float
        Time window over which to marginalise the 4-D coalescence function.
    min_event_interval : float
        Minimum time interval between triggered events.
    normalise_coalescence : bool
        If True, use coalescence normalised by the average coalescence value in the 3-D
        grid at each timestep.

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
        ax.text(
            0.42,
            0.15,
            f"Triggered {count} event(s) on the {trig} trace.",
            ha="center",
            va="center",
        )
    ax.set_axis_off()


def _plot_trigger_region(axes, region):
    """
    Utility function for plotting the bounding geographical box used to filter triggered
    events.

    Parameters
    ----------
    axes : list of `matplotlib.Axes` objects
        Axes on which to plot the bounding boxes.
    region : list
        Geographical region within which to trigger earthquakes.

    """

    min_x, min_y, min_z, max_x, max_y, max_z = region

    # Plot on XY
    axes[0].plot(
        [min_x, min_x, max_x, max_x, min_x],
        [min_y, max_y, max_y, min_y, min_y],
        linestyle="--",
        color="#238b45",
        linewidth=1.5,
    )

    # Plot on XZ
    axes[1].plot(
        [min_x, min_x, max_x, max_x, min_x],
        [min_z, max_z, max_z, min_z, min_z],
        linestyle="--",
        color="#238b45",
        linewidth=1.5,
    )

    # Plot on YZ
    axes[2].plot(
        [min_z, max_z, max_z, min_z, min_z],
        [min_y, min_y, max_y, max_y, min_y],
        linestyle="--",
        color="#238b45",
        linewidth=1.5,
    )


def _plot_xy_files(xy_files, ax):
    """
    Plot xy files supplied by user.

    The user can specify a list of xy files to plot by supplying a csv file with
    columns: ["File", "Color", "Linewidth", "Linestyle"], where "File" is the absolute
    path to the file containing the coordinates to be plotted.
    E.g: "/home/user/volcano_outlines.csv,black,0.5,-"

    Each specified xy file should contain coordinates only, with columns:
    ["Longitude", "Latitude"]. E.g.: "-17.5,64.8".

    Lines pre-pended with `#` will be treated as a comment - this can be used to include
    references. See the Volcanotectonic_Iceland example XY_files for a template.\n

    .. note:: Do not include a header line in either file.

    Parameters
    ----------
    xy_files : str
        Path to .csv file containing a list of coordinates files to plot, and the
        linecolor and style to plot them with.
    ax : `matplotlib.Axes` object
        Axes on which to plot the xy files.

    """

    if xy_files is not None:
        xy_files = pd.read_csv(
            xy_files,
            names=["File", "Color", "Linewidth", "Linestyle"],
            header=None,
            comment="#",
        )
        for _, f in xy_files.iterrows():
            xy_file = pd.read_csv(
                f["File"], names=["Longitude", "Latitude"], header=None, comment="#"
            )
            ax.plot(
                xy_file["Longitude"],
                xy_file["Latitude"],
                ls=f["Linestyle"],
                lw=f["Linewidth"],
                c=f["Color"],
            )

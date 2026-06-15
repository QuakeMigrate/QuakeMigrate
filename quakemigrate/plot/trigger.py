"""
Module to plot the triggered events on a decimated grid.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import quakemigrate.util as util
from quakemigrate.exceptions import NoStationAvailabilityData
from quakemigrate.io import read_availability
from quakemigrate.plot.maps import (
    adjust_map_cross_sections,
    build_2d_map_axes,
    build_3d_map_axes,
    MapAxes2D,
    MapAxes3D,
    plot_stations,
    plot_map_overlays,
)


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from obspy import UTCDateTime

    from quakemigrate.io.core import Run
    from quakemigrate.lut import LUT


@dataclass
class TriggerSummaryAxes3D:
    """Named axes container for a 3-D trigger summary figure."""

    text_summary: Axes
    lut_map: MapAxes3D
    coalescence: Axes
    norm_coalescence: Axes
    availability: Axes


def _setup_axes_3d(fig: Figure, lut: LUT) -> TriggerSummaryAxes3D:
    """
    Build the axes layout for the 3-D trigger summary figure.

    Parameters
    ----------
    fig:
        Figure on which the axes are created.
    lut:
        Lookup table object providing the map geometry and bounds used for the map and
        cross-section panels.

    Returns
    -------
    axes:
        Named collection of axes for the trigger summary, including panels for the text
        summary, map/cross-sections, coalescence traces, and station availability.

    """

    grid_dimensions = (9, 18)
    gs = GridSpec(*grid_dimensions)

    text_summary = fig.add_subplot(gs[0:2, 0:8])
    coalescence = fig.add_subplot(gs[0:3, 8:18])
    norm_coalescence = fig.add_subplot(gs[3:6, 8:18])
    availability = fig.add_subplot(gs[6:9, 8:18])
    for ax in [coalescence, norm_coalescence]:
        ax.sharex(availability)

    lut_axes = build_3d_map_axes(fig, grid_dimensions, lut, "black")

    return TriggerSummaryAxes3D(
        text_summary=text_summary,
        lut_map=lut_axes,
        coalescence=coalescence,
        norm_coalescence=norm_coalescence,
        availability=availability,
    )


@dataclass
class TriggerSummaryAxes2D:
    """Named axes container for a 2-D trigger summary figure."""

    text_summary: Axes
    lut_map: MapAxes2D
    coalescence: Axes
    norm_coalescence: Axes
    availability: Axes


def _setup_axes_2d(fig: Figure, lut: LUT) -> TriggerSummaryAxes2D:
    """
    Build the axes layout for the 2-D trigger summary figure.

    Parameters
    ----------
    fig:
        Figure on which the axes are created.
    lut:
        Lookup table object providing the map geometry and bounds used for the map.

    Returns
    -------
    axes:
        Named collection of axes for the trigger summary, including panels for the text
        summary, map, coalescence traces, and station availability.

    """

    grid_dimensions = (9, 18)
    gs = GridSpec(*grid_dimensions)

    text_summary = fig.add_subplot(gs[0:2, 0:8])
    coalescence = fig.add_subplot(gs[0:3, 8:18])
    norm_coalescence = fig.add_subplot(gs[3:6, 8:18])
    availability = fig.add_subplot(gs[6:9, 8:18])
    for ax in [coalescence, norm_coalescence]:
        ax.sharex(availability)

    lut_axes = build_2d_map_axes(fig, grid_dimensions, lut, "black")

    return TriggerSummaryAxes2D(
        text_summary=text_summary,
        lut_map=lut_axes,
        coalescence=coalescence,
        norm_coalescence=norm_coalescence,
        availability=availability,
    )


@util.timeit("info")
def trigger_summary(
    events: pd.DataFrame,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    run: Run,
    marginal_window: float,
    min_event_interval: float,
    detection_threshold: np.ndarray,
    threshold_string: str,
    normalise_coalescence: bool,
    lut: LUT,
    data: pd.DataFrame,
    region: list,
    discarded_events: pd.DataFrame,
    interactive: bool,
    xy_files: str | None = None,
    plot_all_stations: bool = True,
    file_type: str = "pdf",
) -> None:
    """
    Plots the data from a .scanmseed file with annotations illustrating the trigger
    results: event triggers and marginal windows on the coalescence traces, and map (and
    cross-section view for 3-D case) of the gridded triggered earthquake locations.

    Parameters
    ----------
    events:
        Triggered events information, columns: ["EventID", "CoaTime", "TRIG_COA",
        "COA_X", "COA_Y", "COA_Z", "MinTime", "MaxTime", "COA", "COA_NORM"].
    starttime:
        Start time of trigger run.
    endtime:
        End time of trigger run.
    run:
        Light class encapsulating i/o path information for a given run.
    marginal_window:
        Time window over which to marginalise the 4-D coalescence function.
    min_event_interval:
        Minimum time interval between triggered events.
    detection_threshold:
        Coalescence value above which to trigger events.
    threshold_string:
        String describing the threshold method and parameters used.
    normalise_coalescence:
        If True, use coalescence normalised by the average coalescence value in the grid
        at each timestep.
    lut:
        Contains the traveltime lookup tables for the selected seismic phases, computed
        for some pre-defined velocity model.
    data:
        Data output by :func:`~quakemigrate.signal.scan.QuakeScan.detect()` --
        continuous scan, columns: ["DT", "COA", "COA_N", "X", "Y", "Z"]
    region:
        Geographical region within which to trigger earthquakes; events located outside
        this region will be discarded.
    discarded_events:
        Discarded triggered events information, columns: ["EventID", "CoaTime",
        "TRIG_COA", "COA_X", "COA_Y", "COA_Z", "MinTime", "MaxTime", "COA", "COA_NORM"].
    interactive:
        Toggles whether to produce an interactive plot.
    xy_files:
        Path to comma-separated value file (.csv) containing a series of coordinate
        files to plot. Columns: ["File", "Color", "Linewidth", "Linestyle"], where
        "File" is the absolute path to the file containing the coordinates to be
        plotted. E.g: "/home/user/volcano_outlines.csv,black,0.5,-". Each .csv
        coordinate file should contain coordinates only, with columns: ["Longitude",
        "Latitude"]. E.g.: "-17.5,64.8". Lines pre-pended with # will be treated as
        a comment - this can be used to include references. See the
        Volcanotectonic_Iceland example XY_files for a template.\n
        .. note:: Do not include a header line in either file.
    plot_all_stations:
        If true, plot all stations used for detect. Otherwise, only plot stations which
        for which some data was available during the trigger time window. NOTE: if no
        station availability data is found, all stations in the LUT will be plotted.
    file_type:
        File format to use for output.

    """

    dt = pd.to_datetime(data["DT"].astype(str)).values

    fig = plt.figure(figsize=(30, 15))
    if lut.node_count[2] == 1:
        axes = _setup_axes_2d(fig, lut)
    else:
        axes = _setup_axes_3d(fig, lut)

    logging.debug(discarded_events)

    # --- Write summary information ---
    _plot_text_summary(
        axes.text_summary,
        starttime,
        endtime,
        events,
        threshold_string,
        marginal_window,
        min_event_interval,
        normalise_coalescence,
    )

    # --- Plot LUT, coalescence traces, and station availability ---
    _plot_coalescence(axes.coalescence, dt, data.COA.values, "Maximum coalescence")
    _plot_coalescence(
        axes.norm_coalescence, dt, data.COA_N.values, "Normalised maximum coalescence"
    )
    try:
        availability = read_availability(run, starttime, endtime)
        _plot_station_availability(axes.availability, availability, endtime)
    except NoStationAvailabilityData as e:
        logging.info(e)
        availability = None

    # --- Add trigger threshold to the correct coalescence trace ---
    ax_i = axes.norm_coalescence if normalise_coalescence else axes.coalescence
    ax_i.step(dt, detection_threshold, where="mid", c="g", label="Detection threshold")

    for ax in [axes.coalescence, axes.norm_coalescence, axes.availability]:
        ax.set_xlim([starttime.datetime, endtime.datetime])

    # --- Plot trigger region (if any) ---
    if region is not None:
        _plot_trigger_region(axes.lut_map, region)
        _plot_event_windows(
            [axes.coalescence, axes.norm_coalescence],
            discarded_events,
            marginal_window,
            discarded=True,
        )
        _plot_event_locations(axes.lut_map, discarded_events, discarded=True)

    # --- Plot event scatter on LUT and windows on coalescence traces ---
    if not events.empty:
        _plot_event_windows(
            [axes.coalescence, axes.norm_coalescence], events, marginal_window
        )
        _plot_event_locations(axes.lut_map, events)

    # Use station availability to work out which stations to plot
    if availability is None:
        stations = lut.stations
    elif plot_all_stations:
        station_names = {key.split("_")[0] for key in availability.columns}
        stations = [station for station in lut.stations if station.id in station_names]
    else:
        station_names = {
            key.split("_")[0]
            for key, available in availability.items()
            if available == 1
        }
        stations = [station for station in lut.stations if station.id in station_names]
    plot_stations(axes.lut_map, stations, "k")

    if xy_files is not None:
        plot_map_overlays(xy_files, axes.lut_map.xy)

    # --- Handle legend for coalescence trace plot ---
    handles, labels = ax_i.get_legend_handles_labels()
    uniq_labels = dict(zip(labels, handles))
    ax_i.legend(
        uniq_labels.values(), uniq_labels.keys(), loc=1, fontsize=14, framealpha=0.85
    ).set_zorder(20)

    fig.tight_layout(pad=1, h_pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.canvas.draw()

    if isinstance(axes.lut_map, MapAxes3D):
        adjust_map_cross_sections(fig, axes.lut_map)

    fpath = run.path / "trigger" / run.subname / "summaries"
    fpath.mkdir(exist_ok=True, parents=True)
    fstem = f"{run.name}_{starttime.year}_{starttime.julday:03d}_Trigger"
    file = (fpath / fstem).with_suffix(f".{file_type}")
    fig.savefig(file, dpi=400)

    if interactive:
        plt.show()

    plt.close(fig)


def _plot_station_availability(
    ax: Axes, availability: pd.DataFrame, endtime: UTCDateTime
) -> None:
    """
    Plot station availability through time.

    Parameters
    ----------
    ax:
        Axes on which to plot the station availability.
    availability:
        Dataframe containing station/phase availability through time.
    endtime:
        End time of the trigger run, used to close the final step in the plot.

    """

    # Get list of phases from station availability dataframe
    phases = sorted(set([col_name.split("_")[1] for col_name in availability.columns]))
    logging.debug(f"\t\t    Found phases: {phases}")

    # Sort out plotting options based on the number of phases
    if len(phases) > 2:
        logging.warning(
            "\t\t    Only P and/or S are currently supported! Plotting by station only."
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
            # This can lead to incorrect value (e.g., if 2 / 3 phases are
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


def _plot_coalescence(ax: Axes, dt: np.ndarray, data: np.ndarray, label: str) -> None:
    """
    Plot a coalescence trace through time.

    Parameters
    ----------
    ax:
        Axes on which to plot the trace.
    dt:
        Timestamps corresponding to the coalescence samples.
    data:
        Coalescence values to plot.
    label:
        Label used for the y-axis and panel tag.

    """

    ax.plot(dt, data, c="k", lw=0.01, label="Coalescence value", alpha=0.8, zorder=10)
    _add_plot_tag(ax, label)
    ax.set_ylabel(label, fontsize=14)
    ax.xaxis.set_major_formatter(util.DateFormatter("%H:%M:%S.{ms}", 2))


def _add_plot_tag(ax: Axes, tag: str) -> None:
    """
    Add a descriptive tag box to a plotted data panel.

    Parameters
    ----------
    ax:
        Axes on which to add the tag.
    tag:
        Text to display in the tag box.

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


def _plot_event_locations(
    axes: MapAxes2D | MapAxes3D, events: pd.DataFrame, discarded: bool = False
) -> None:
    """
    Plot triggered-event locations on the map and cross-section panels.

    Parameters
    ----------
    axes:
        Map and cross-section axes on which to plot event locations.
    events:
        Dataframe of triggered events.
    discarded:
        Whether supplied events are discarded (due to being outside the trigger region,
        or outside the trigger time window).

    """

    x, y, z = events[["COA_X", "COA_Y", "COA_Z"]].values.T

    if discarded:
        axes.xy.scatter(x, y, s=50, c="grey")
        if isinstance(axes, MapAxes3D):
            axes.xz.scatter(x, z, s=50, c="grey")
            axes.yz.scatter(z, y, s=50, c="grey")

    else:
        # Get bounds for cmap - hack to prevent inconsistent color being
        # assigned when only a single event has been triggered.
        vmin, vmax = (
            events["TRIG_COA"].min() * 0.999,
            events["TRIG_COA"].max() * 1.001,
        )

        # Plotting the scatter of the earthquake locations
        c = events["TRIG_COA"].values
        sc = axes.xy.scatter(x, y, s=50, c=c, vmin=vmin, vmax=vmax)

        if isinstance(axes, MapAxes3D):
            axes.xz.scatter(x, z, s=50, c=c, vmin=vmin, vmax=vmax)
            axes.yz.scatter(z, y, s=50, c=c, vmin=vmin, vmax=vmax)

        cb = axes.cax.figure.colorbar(
            sc, ax=axes.cax, orientation="horizontal", fraction=0.8, aspect=8
        )
        cb.ax.set_xlabel("Peak coalescence value", rotation=0, fontsize=14)


def _plot_event_windows(
    axes: list[Axes],
    events: pd.DataFrame,
    marginal_window: float,
    discarded: bool = False,
) -> None:
    """
    Plot trigger windows (marginal window and minimum event interval) on coalescence
    traces.

    Parameters
    ----------
    axes:
        Coalescence-trace axes on which to draw the windows.
    events:
        Dataframe of triggered events.
    marginal_window:
        Half-width of the marginal window, in seconds.
    discarded:
        If True, plot discarded-event windows using the discarded-event style.

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
    ax: Axes,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    events: pd.DataFrame,
    threshold_string: str,
    marginal_window: float,
    min_event_interval: float,
    normalise_coalescence: bool,
) -> None:
    """
    Add text summary of triggered events.

    Parameters
    ----------
    ax:
        Axes on which to plot the text summary.
    starttime:
        Start time of trigger run.
    endtime:
        End time of trigger run.
    events:
        DataFrame of retained triggered events.
    threshold_string:
        String describing the threshold method and parameters used.
    marginal_window:
        Time window over which to marginalise the 4-D coalescence function.
    min_event_interval:
        Minimum time interval between triggered events.
    normalise_coalescence:
        If True, use coalescence normalised by the average coalescence value in the 3-D
        grid at each timestep.

    """

    st, et = [t.strftime("%Y-%m-%d %H:%M:%S") for t in (starttime, endtime)]
    ax.text(0.42, 0.8, f"{st}  -  {et}", fontsize=20, fontweight="bold", ha="center")

    # Get trigger on and event count info
    trig = "normalised coalescence" if normalise_coalescence else "coalescence"
    count = len(events)

    with plt.rc_context({"font.size": 18}):
        ax.text(0.45, 0.65, "Trigger threshold:", ha="right", va="center")
        ax.text(0.47, 0.65, f"{threshold_string}", ha="left", va="center")
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


def _plot_trigger_region(axes: MapAxes2D | MapAxes3D, region: list) -> None:
    """
    Plot the geographic bounding box used to filter triggered events.

    Parameters
    ----------
    axes:
        Map and cross-section axes on which to draw the region bounds.
    region:
        Geographic bounding region used for filtering triggered events, given as lower
        and upper bounds in the coordinate system of the plotted panels.

    """

    for _, ax, (i, j), _ in axes.items():
        ax.plot(
            [region[i], region[i], region[i + 3], region[i + 3], region[i]],
            [region[j], region[j + 3], region[j + 3], region[j], region[j]],
        )

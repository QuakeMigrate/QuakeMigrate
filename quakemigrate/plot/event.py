"""
Tools for generating 2-D and 3-D event summary visualisations.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

import quakemigrate.util as util
from quakemigrate.plot.maps import (
    adjust_map_cross_sections,
    build_2d_map_axes,
    build_3d_map_axes,
    plot_map_overlays,
    plot_stations,
)
from quakemigrate.signal.location_uncertainty import embed_matrix


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from quakemigrate.io.core import Run
    from quakemigrate.io.event import Event
    from quakemigrate.lut import LUT
    from quakemigrate.plot.maps import MapAxes2D, MapAxes3D


@dataclass
class EventSummaryAxes2D:
    """Named axes container for a 2-D event summary plot."""

    text_summary: Axes
    coalescence_map: MapAxes2D
    waveform_gather: Axes
    coalescence_timeseries: Axes


def _setup_axes_2d(fig: Figure, lut: LUT) -> EventSummaryAxes2D:
    """
    Build the axes layout for the standard 2-D event summary figure.

    Parameters
    ----------
    fig:
        Figure on which the axes are created.
    lut:
        Lookup table object providing the map geometry and bounds.

    Returns
    -------
    axes:
        Named collection of axes for the 2-D event summary, including panels for the
        text summary, map, waveform gather, and coalescence time series.

    """

    grid_dimensions = (9, 15)
    gs = GridSpec(*grid_dimensions)

    text_summary = fig.add_subplot(gs[0:2, 0:8])
    waveform_gather = fig.add_subplot(gs[0:7, 8:15])
    coalescence_timeseries = fig.add_subplot(gs[7:9, 8:15])

    lut_axes = build_2d_map_axes(fig, grid_dimensions, lut, "white")

    return EventSummaryAxes2D(
        text_summary=text_summary,
        coalescence_map=lut_axes,
        waveform_gather=waveform_gather,
        coalescence_timeseries=coalescence_timeseries,
    )


def _extract_2d_coalescence_slices(
    marginalised_coa_map: np.ndarray[np.double],
    lut: LUT,
    slice_mode: Literal["surface", "maximum"] = "surface",
    surface_depth: float = 0.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Extract a 2-D XY slice from a 3-D marginalised coalescence map.

    Parameters
    ----------
    marginalised_coa_map:
        Marginalised 3-D coalescence map.
    lut:
        Lookup table object used to convert the requested surface depth to a grid index.
    slice_mode:
        Strategy used to select the XY slice. ``"surface"`` extracts the slice at
        ``surface_depth``. ``"maximum"`` extracts the slice through the depth index of
        the global 3-D coalescence maximum.
    surface_depth:
        Depth, in the LUT coordinate system, used when ``slice_mode="surface"``.

    Returns
    -------
    idx_max:
        Grid index of the reference maximum used for plotting and traveltime evaluation.
    slices:
        Dictionary containing the extracted XY coalescence slice.

    Raises
    ------
    ValueError
        If an invalid ``slice_mode`` is supplied.

    """

    coa_map = np.ma.masked_invalid(marginalised_coa_map)
    if slice_mode == "surface":
        surface_idx = lut.index2grid(
            [lut.ll_corner[0], lut.ll_corner[1], surface_depth],
            inverse=True,
        )[0][2]
        surface_slice = coa_map[:, :, surface_idx]
        idx_max_2d = np.unravel_index(np.nanargmax(surface_slice), surface_slice.shape)
        idx_max = np.array([idx_max_2d[0], idx_max_2d[1], surface_idx])
    elif slice_mode == "maximum":
        idx_max = np.column_stack(np.where(coa_map == np.nanmax(coa_map)))[0]
        surface_slice = coa_map[:, :, idx_max[2]]
    else:
        raise ValueError(
            f"Invalid slice_mode {slice_mode}. Expected 'surface' or 'maximum'."
        )

    slices = {"xy": surface_slice}

    return idx_max, slices


@dataclass
class EventSummaryAxes3D:
    """Named axes container for a 3-D event summary plot."""

    text_summary: Axes
    coalescence_map: MapAxes3D
    waveform_gather: Axes
    coalescence_timeseries: Axes


def _setup_axes_3d(fig: Figure, lut: LUT) -> EventSummaryAxes3D:
    """
    Build the axes layout for the standard 3-D event summary figure.

    Parameters
    ----------
    fig:
        Figure on which the axes are created.
    lut:
        Lookup table object providing the map geometry and bounds.

    Returns
    -------
    axes:
        Named collection of axes for the 3-D event summary, including panels for the
        text summary, map/cross-sections, waveform gather, and coalescence time series.

    """

    grid_dimensions = (9, 15)
    gs = GridSpec(*grid_dimensions)

    text_summary = fig.add_subplot(gs[0:2, 0:8])
    waveform_gather = fig.add_subplot(gs[0:7, 8:15])
    coalescence_timeseries = fig.add_subplot(gs[7:9, 8:15])

    lut_axes = build_3d_map_axes(fig, grid_dimensions, lut, "white")

    return EventSummaryAxes3D(
        text_summary=text_summary,
        coalescence_map=lut_axes,
        waveform_gather=waveform_gather,
        coalescence_timeseries=coalescence_timeseries,
    )


def _extract_3d_coalescence_slices(
    marginalised_coa_map: np.ndarray[np.double],
) -> tuple[tuple[int, int, int], dict[str, np.ndarray]]:
    """
    Extract orthogonal slices through the maximum of a 3-D coalescence map.

    Parameters
    ----------
    marginalised_coa_map:
        Marginalised 3-D coalescence map.

    Returns
    -------
    idx_max:
        Grid index of the maximum coalescence value.
    slices:
        Dictionary containing orthogonal coalescence slices.

    """

    coa_map = np.ma.masked_invalid(marginalised_coa_map)
    idx_max = np.column_stack(np.where(coa_map == np.nanmax(coa_map)))[0]

    slices = {
        "xy": coa_map[:, :, idx_max[2]],
        "xz": coa_map[:, idx_max[1], :],
        "yz": coa_map[idx_max[0], :, :].T,
    }

    return idx_max, slices


def _merge_waveform_legend_labels(
    handles: list[str],
    labels: list[str],
    channel_maps: dict[str, str],
) -> tuple[list, list]:
    """
    Merge duplicate waveform legend entries for equivalent component mappings.

    Parameters
    ----------
    handles:
        Matplotlib legend handles.
    labels:
        Legend labels corresponding to ``handles``.
    channel_maps:
        Mapping from phase name to the configured component string used for that phase.

    Returns
    -------
    handles:
        Deduplicated legend handles.
    labels:
        Deduplicated legend labels, with equivalent component entries merged where
        possible.

    """

    merged_labels = list(labels)

    for phase, component_string in channel_maps.items():
        components = component_string.strip("[]").split(",")
        components = [c.strip() for c in components if c.strip()]

        if len(components) < 2:
            continue

        # pair successive components: [N,1,E,2] -> [(N,1), (E,2)]
        for i in range(0, len(components) - 1, 2):
            cp1, cp2 = components[i], components[i + 1]

            label1 = f"{cp1} component ({phase})"
            label2 = f"{cp2} component ({phase})"

            if label1 in merged_labels and label2 in merged_labels:
                merged_labels = [
                    f"{cp2}, {cp1} component ({phase})"
                    if x == label1 or x == label2
                    else x
                    for x in merged_labels
                ]

    by_label = dict(zip(merged_labels, handles))

    return list(by_label.values()), list(by_label.keys())


@util.timeit("info")
def event_summary_3d(
    run: Run,
    event: Event,
    marginalised_coa_map: np.ndarray[np.double],
    lut: LUT,
    overlay_manifest: str | None = None,
    plot_all_stations: bool = True,
    file_type: str = "pdf",
) -> None:
    """
    Plot a 3-D event summary figure.

    The figure includes:
        - orthogonal slices through the marginalised 3-D coalescence map
        - event location and uncertainty ellipses on the map/cross-section panels
        - a waveform gather of the pre-processed traces used to calculate onset
          functions (sorted by distance from the event)
        - and the maximum coalescence trace through time

    Parameters
    ----------
    run:
        Light class encapsulating i/o path information for a given run.
    event:
        Light class encapsulating waveforms, coalescence information, picks and
        location information for a given event.
    marginalised_coa_map:
        Marginalised 3-D coalescence map, shape(nx, ny, nz).
    lut:
        Traveltime lookup table object describing the spatial grid and geometry.
    overlay_manifest:
        Path to a map-overlay manifest file describing one or more overlays to draw on
        the XY map panel.
    plot_all_stations:
        If True, plot all stations. Otherwise, plot only stations for which data were
        available for the event.
    file_type:
        Output file format for saved figure.

    """

    logging.info("\tPlotting 3-D event summary figure...")
    fig = plt.figure(figsize=(25, 15))
    axes = _setup_axes_3d(fig, lut)

    # --- Write summary information ---
    _plot_text_summary(axes.text_summary, lut, event)

    # --- Plot slices through 3-D coalescence ---
    idx_max, slices = _extract_3d_coalescence_slices(marginalised_coa_map)
    _plot_coalescence_panels(axes.coalescence_map, slices)

    if plot_all_stations:
        station_list = event.data.stations
    else:
        station_list = {
            key.split("_")[0]
            for key, available in event.onset_data.availability.items()
            if available == 1
        }
    station_data = lut.station_data[lut.station_data["Name"].isin(station_list)]
    plot_stations(axes.coalescence_map, station_data, "white")

    # Add hypocentre and Gaussian uncertainty ellipses
    _plot_hypocentre(axes.coalescence_map, hypocentre=event.hypocentre)
    gues = _make_ellipses(lut, event, "gaussian", "k")
    for (_, ax, *_), gue in zip(axes.coalescence_map.items(), gues):
        ax.add_patch(gue)

    if overlay_manifest is not None:
        plot_map_overlays(overlay_manifest, axes.coalescence_map.xy)

    axes.coalescence_map.xy.legend(fontsize=14)

    # --- Plot waveform gather ---
    _plot_waveform_gather(axes.waveform_gather, lut, event, idx_max, station_list)

    # --- Plot 1-D maximum coalescence time series ---
    _plot_coalescence_trace(axes.coalescence_timeseries, event)

    # --- Add event origin time to waveform gather and coalescence plots ---
    for ax in [axes.waveform_gather, axes.coalescence_timeseries]:
        ax.axvline(
            event.otime.datetime, label="Origin time", ls="--", lw=2, c="#F03B20"
        )

    handles, labels = axes.waveform_gather.get_legend_handles_labels()
    handles, labels = _merge_waveform_legend_labels(
        handles,
        labels,
        event.onset_data.channel_maps,
    )
    axes.waveform_gather.legend(
        handles,
        labels,
        fontsize=14,
        loc=1,
        framealpha=1,
        markerscale=0.5,
    )
    axes.coalescence_timeseries.legend(fontsize=14, loc=1, framealpha=1)

    fig.tight_layout(pad=1, h_pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.canvas.draw()
    adjust_map_cross_sections(fig, axes.coalescence_map)

    fpath = run.path / "locate" / run.subname / "summaries"
    fpath.mkdir(exist_ok=True, parents=True)
    fstem = f"{run.name}_{event.uid}_EventSummary3D"
    file = (fpath / fstem).with_suffix(f".{file_type}")
    fig.savefig(file, dpi=400)
    plt.close(fig)


event_summary = event_summary_3d


@util.timeit("info")
def event_summary_2d(
    run: Run,
    event: Event,
    marginalised_coa_map: np.ndarray[np.double],
    lut: LUT,
    slice_mode: Literal["surface", "maximum"] = "surface",
    surface_depth: float = 0.0,
    overlay_manifest: str | None = None,
    plot_all_stations: bool = True,
    file_type: str = "pdf",
):
    """
    Plot a 2-D event summary figure.

    The figure includes:
        - ax XY slice through the marginalised coalescence map
        - event location and uncertainty ellipses on the map panel
        - a waveform gather of the pre-processed traces used to calculate onset
          functions (sorted by distance from the event)
        - and the maximum coalescence trace through time

    Parameters
    ----------
    run:
        Light class encapsulating i/o path information for a given run.
    event:
        Light class encapsulating waveforms, coalescence information, picks and
        location information for a given event.
    marginalised_coa_map:
        Marginalised 3-D coalescence map, shape(nx, ny, nz).
    lut:
        Traveltime lookup table object describing the spatial grid and geometry.
    slice_mode:
        Strategy used to select the XY slice. ``"surface"`` extracts the slice at
        ``surface_depth``. ``"maximum"`` extracts the slice through the depth index of
        the global 3-D coalescence maximum.
    surface_depth:
        Depth, in the LUT coordinate system, used when ``slice_mode="surface"``.
    overlay_manifest:
        Path to a map-overlay manifest file describing one or more overlays to draw on
        the XY map panel.
    plot_all_stations:
        If True, plot all stations. Otherwise, plot only stations for which data were
        available for the event.
    file_type:
        Output file format for saved figure.

    """

    logging.info("\tPlotting event summary figure...")
    fig = plt.figure(figsize=(25, 15))
    axes = _setup_axes_2d(fig, lut)

    # --- Write summary information ---
    _plot_text_summary(axes.text_summary, lut, event)

    # Extract indices and grid coordinates of maximum coalescence
    idx_max, slices = _extract_2d_coalescence_slices(
        marginalised_coa_map, lut, slice_mode, surface_depth
    )
    _plot_coalescence_panels(axes.coalescence_map, slices)

    if plot_all_stations:
        station_list = event.data.stations
    else:
        station_list = {
            key.split("_")[0]
            for key, available in event.onset_data.availability.items()
            if available == 1
        }
    station_data = lut.station_data[lut.station_data["Name"].isin(station_list)]
    plot_stations(axes.coalescence_map, station_data, "white")

    # Add hypocentre and Gaussian uncertainty ellipses
    _plot_hypocentre(axes.coalescence_map, hypocentre=event.hypocentre)
    gaussian_ellipses = _make_ellipses(lut, event, "gaussian", "k")
    for (_, ax, *_), ellipse in zip(axes.coalescence_map.items(), gaussian_ellipses):
        if ellipse is not None:
            ax.add_patch(ellipse)

    if overlay_manifest is not None:
        plot_map_overlays(overlay_manifest, axes.coalescence_map.xy)

    axes.coalescence_map.xy.legend(fontsize=14)

    # --- Plot waveform gather ---
    _plot_waveform_gather(axes.waveform_gather, lut, event, idx_max, station_list)

    # --- Plot 1-D maximum coalescence time series ---
    _plot_coalescence_trace(axes.coalescence_timeseries, event)

    # --- Add event origin time to waveform gather and coalescence plots ---
    for ax in [axes.waveform_gather, axes.coalescence_timeseries]:
        ax.axvline(
            event.otime.datetime, label="Origin time", ls="--", lw=2, c="#F03B20"
        )

    handles, labels = axes.waveform_gather.get_legend_handles_labels()
    handles, labels = _merge_waveform_legend_labels(
        handles,
        labels,
        event.onset_data.channel_maps,
    )
    axes.waveform_gather.legend(
        handles,
        labels,
        fontsize=14,
        loc=1,
        framealpha=1,
        markerscale=0.5,
    )
    axes.coalescence_timeseries.legend(fontsize=14, loc=1, framealpha=1)

    fig.tight_layout(pad=1, h_pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.canvas.draw()

    fpath = run.path / "locate" / run.subname / "summaries"
    fpath.mkdir(exist_ok=True, parents=True)
    fstem = f"{run.name}_{event.uid}_EventSummary2D"
    file = (fpath / fstem).with_suffix(f".{file_type}")
    fig.savefig(file, dpi=400)
    plt.close(fig)


def _plot_coalescence_panels(
    axes: MapAxes2D | MapAxes3D, slices: dict[str, np.ndarray]
):
    """
    Plot coalescence slices on map and cross-section panels.

    Parameters
    ----------
    axes:
        Map and cross-section axes on which to draw the coalescence map.
    slices:
        Dictionary of coalescence slices keyed by panel name (for example
        ``"xy"``, ``"xz"``, and ``"yz"``).

    """

    for ax_label, ax, (i, j), _ in axes.items():
        gminx, gmaxx = axes.bounds[i]
        gminy, gmaxy = axes.bounds[j]

        slice_ = slices[ax_label]
        nx, ny = [dim + 1 for dim in slice_.shape]
        grid1, grid2 = np.mgrid[gminx : gmaxx : nx * 1j, gminy : gmaxy : ny * 1j]
        sc = ax.pcolormesh(grid1, grid2, slice_, edgecolors="face")

        if ax_label != "xy":
            continue

        # --- Add colourbar ---
        cb = axes.cax.figure.colorbar(
            sc, ax=axes.cax, orientation="horizontal", fraction=0.8, aspect=8
        )
        cb.ax.set_xlabel("Normalised coalescence\nvalue", rotation=0, fontsize=14)


def _plot_hypocentre(axes: MapAxes2D | MapAxes3D, hypocentre):
    """
    Plot hypocentre crosshairs on map and cross-section panels.

    Parameters
    ----------
    axes:
        Map and cross-section axes on which to draw the hypocentre.
    hypocentre:
        Event hypocentre coordinates in the LUT coordinate system.

    """

    for _, ax, (i, j), _ in axes.items():
        ax.axvline(x=hypocentre[i], ls="--", lw=1.5, c="white")
        ax.axhline(y=hypocentre[j], ls="--", lw=1.5, c="white")


def _plot_text_summary(ax: Axes, lut: LUT, event: Event) -> None:
    """
    Plot the textual event summary panel.

    Parameters
    ----------
    ax:
        Axes on which to plot the text summary.
    lut:
        Traveltime lookup table object used to format location values and units.
    event:
        Event object containing origin time, location, uncertainties, and optional
        magnitude information.

    """

    # Grab a conversion factor based on the grid projection to convert the
    # hypocentre depth + uncertainties to the correct units and evaluate the
    # suitable precision to which to report results from the LUT.
    km_cf = 1000 / lut.unit_conversion_factor
    precision = [max((prec + 2), 6) for prec in lut.precision[:2]]
    unit_correction = 3 if lut.unit_name == "km" else 0
    precision.append(max((lut.precision[2] + 2), 0 + unit_correction))

    hypocentre = [round(dimh, dimp) for dimh, dimp in zip(event.hypocentre, precision)]
    gau_unc = [round(dim, precision[2]) for dim in event.loc_uncertainty / km_cf]
    hypo = (
        f"{hypocentre[1]}\u00b0N \u00b1 {gau_unc[1]} km\n"
        f"{hypocentre[0]}\u00b0E \u00b1 {gau_unc[0]} km\n"
        f"{hypocentre[2] / km_cf} \u00b1 {gau_unc[2]} km"
    )

    # Grab the covariance error and magnitude information
    cov_err_xyz = event.locations["covariance"]["Err_XYZ"]
    mag_info = event.local_magnitude

    ax.text(0.25, 0.8, f"Event: {event.uid}", fontsize=20, fontweight="bold")
    ot_text = event.otime.strftime("%Y-%m-%d %H:%M:%S.")
    ot_text += event.otime.strftime("%f")[:3]
    with plt.rc_context({"font.size": 16}):
        ax.text(0.35, 0.65, "Origin time:", ha="right", va="center")
        ax.text(0.37, 0.65, f"{ot_text}", ha="left", va="center")
        ax.text(0.35, 0.55, "Hypocentre:", ha="right", va="top")
        ax.text(0.37, 0.55, hypo, ha="left", va="top")
        ax.text(0.35, 0.22, "Geometric mean covariance:", ha="right", va="center")
        ax.text(0.37, 0.22, f"{cov_err_xyz:.3g}", ha="left", va="center")
        if mag_info is not None:
            mag, mag_err, mag_r2 = mag_info
            ax.text(0.35, 0.09, "Local magnitude:", ha="right")
            ax.text(
                0.37,
                0.09,
                f"{mag:.3g} \u00b1 {mag_err:.3g}   r\u00b2 = {mag_r2:.3g}",
                ha="left",
            )
    ax.set_axis_off()


WAVEFORM_COLOURS1 = ["#FB9A99", "#7570b3", "#1b9e77"]
WAVEFORM_COLOURS2 = ["#33a02c", "#b2df8a", "#1f78b4"]
PICK_COLOURS = ["#F03B20", "#3182BD"]


def _plot_waveform_gather(
    ax: Axes,
    lut: LUT,
    event: Event,
    idx_max: np.ndarray[np.int64],
    stations: str | list[str],
) -> None:
    """
    Plot the waveform gather and modelled phase-arrival markers.

    Parameters
    ----------
    ax:
        Axes on which to plot the waveform gather.
    lut:
        Traveltime lookup table object used to serve modelled phase arrivals.
    event:
        Event object containing waveform and onset data.
    idx_max:
        Grid index of the reference location used to calculate modelled traveltimes.
    stations:
        Station name or list of station names for which to plot arrivals and waveforms.

    """

    phases = event.onset_data.phases

    # --- Predicted traveltimes ---
    traveltimes = np.array(
        [lut.traveltime_to(phase, idx_max, stations) for phase in phases]
    )
    arrivals = [[(event.otime + tt).datetime for tt in tt_f] for tt_f in traveltimes]

    range_order = abs(np.argsort(np.argsort(arrivals[0])) - len(arrivals[0])) * 2

    # --- Plot modelled phase arrival times ---
    # estimate the appropriate height for the pick marker line based on the plot height
    s = (ax.get_window_extent().height / (max(range_order) + 1) * 1.2) ** 2

    # Handle single-phase plotting
    pick_colours = PICK_COLOURS
    if len(phases) == 1:
        if phases[0] == "P":
            pick_colours = [PICK_COLOURS[0]]
    for arrival, c, phase in zip(arrivals, pick_colours, phases):
        ax.scatter(
            arrival,
            range_order,
            s=s,
            c=c,
            marker="|",
            zorder=5,
            lw=1.5,
            label=f"Modelled {phase}",
        )

    # --- Waveforms ---
    waveforms = event.onset_data.filtered_waveforms
    p_str, s_str_1, s_str_2 = util.get_phase_component_strings(
        event.onset_data.channel_maps
    )
    # Min and max times to plot
    mint = event.otime - 0.1
    maxt = min(event.otime + np.max(traveltimes) * 1.5, event.data.endtime)
    # Convert to indices -- will still be the same for sub-sample shifts
    times_utc = waveforms[0].times("UTCDateTime")
    mint_i, maxt_i = [np.argmin(abs(times_utc - t)) for t in (mint, maxt)]
    for i, station in enumerate(stations):
        stn_waveforms = waveforms.select(station=station)
        for c, comp, phase in zip(
            WAVEFORM_COLOURS1, [p_str, s_str_1, s_str_2], ["P", "S", "S"]
        ):
            st = stn_waveforms.select(component=comp)
            if not bool(st):
                continue
            # If multiple traces for a given phase, plot both in the same colour
            for tr in st:
                comp = tr.stats.component
                data = tr.data

                # Get station specific range for norm factor
                stat_maxt = event.otime + max(traveltimes[:, i]) * 1.5
                norm = max(abs(data[mint_i : np.argmin(abs(times_utc - stat_maxt))]))

                # Generate times for plotting
                times = tr.times("matplotlib")[mint_i:maxt_i]

                # Trim to plot limits, normalise, shift by range, then plot
                y = data[mint_i:maxt_i] / norm + range_order[i]
                label = f"{comp} component ({phase})"
                ax.plot(times, y, c=c, lw=0.3, label=label, alpha=0.85)

    # --- Limits, annotations, and axis formatting ---
    ax.set_xlim([mint.datetime, maxt.datetime])
    ax.set_ylim([0, max(range_order) + 2])
    ax.xaxis.set_major_formatter(util.DateFormatter("%H:%M:%S.{ms}", 2))
    ax.yaxis.set_ticks(range_order)
    ax.yaxis.set_ticklabels(stations, fontsize=14)


def _plot_coalescence_trace(ax: Axes, event: Event) -> None:
    """
    Plot the maximum coalescence trace around the event origin time.

    Parameters
    ----------
    ax:
        Axes on which to plot the coalescence trace.
    event:
        Event object containing the coalescence time series.

    """

    times = [x.datetime for x in event.coa_data["DT"]]
    ax.plot(
        times,
        event.coa_data["COA"],
        c="k",
        lw=0.5,
        zorder=10,
        label="Maximum coalescence",
    )
    ax.set_ylabel("Maximum coalescence", fontsize=14)
    ax.set_xlabel("DateTime", fontsize=14)
    ax.set_xlim([times[0], times[-1]])
    ax.xaxis.set_major_formatter(util.DateFormatter("%H:%M:%S.{ms}", 2))


def _make_ellipses(
    lut: LUT,
    event: Event,
    uncertainty: Literal["covariance", "gaussian"],
    clr: str,
) -> tuple[Ellipse, Ellipse | None, Ellipse | None]:
    """
    Construct uncertainty ellipses for the map and cross-section panels.

    Parameters
    ----------
    lut:
        Traveltime lookup table object used to convert location uncertainties into
        plotted coordinates.
    event:
        Event object containing hypocentre and uncertainty information.
    uncertainty:
        Uncertainty measure to visualise.
    clr:
        Ellipse edge colour.

    Returns
    -------
    xy, yz, xz:
        Ellipses for the requested uncertainty measure.

    """

    coord = event.get_hypocentre(method="spline")

    if uncertainty == "gaussian":
        gaussian = event.locations.get("gaussian", {})
        covariance_grid = gaussian.get("covariance_matrix")

        if covariance_grid is not None:
            covariance_grid = np.asarray(covariance_grid, dtype=float)

            fit_dims = gaussian.get("fit_dims", [0, 1, 2])

            covariance_coord = _transform_covariance_to_coord_units(
                lut,
                coord,
                covariance_grid,
                fit_dims,
            )

            xy = _ellipse_from_covariance(
                centre=(coord[0], coord[1]),
                covariance_2d=covariance_coord[np.ix_([0, 1], [0, 1])],
                clr=clr,
                label="Gaussian uncertainty",
            )

            xz = _ellipse_from_covariance(
                centre=(coord[0], coord[2]),
                covariance_2d=covariance_coord[np.ix_([0, 2], [0, 2])],
                clr=clr,
            )

            yz = _ellipse_from_covariance(
                centre=(coord[2], coord[1]),
                covariance_2d=covariance_coord[np.ix_([2, 1], [2, 1])],
                clr=clr,
            )

            return xy, xz, yz


def _transform_covariance_to_coord_units(
    lut: LUT,
    grid_idx: np.ndarray,
    covariance_grid: np.ndarray,
    fit_dims: list,
) -> np.ndarray:
    """
    Transform a covariance matrix from grid-index units to LUT coordinate units.

    Parameters
    ----------
    lut:
        Traveltime lookup table object used to convert grid indices to coordinates.
    grid_idx:
        Grid-index location at which to evaluate the local coordinate transform.
    covariance_grid:
        Covariance matrix in grid-index units.
    fit_dims:
        Active dimensions included in the Gaussian fit.

    Returns
    -------
    covariance_coord:
        Covariance matrix transformed into LUT coordinate units.

    """

    covariance_fit = covariance_grid[np.ix_(fit_dims, fit_dims)]

    jacobian = np.zeros((len(fit_dims), len(fit_dims)))
    for col, dim in enumerate(fit_dims):
        step = np.zeros(3)
        step[dim] = 1.0

        coord_plus = lut.index2coord([grid_idx + step])[0]
        coord_minus = lut.index2coord([grid_idx - step])[0]

        jacobian[:, col] = (coord_plus[fit_dims] - coord_minus[fit_dims]) / 2.0

    return embed_matrix(jacobian @ covariance_fit @ jacobian.T, fit_dims)


def _ellipse_from_covariance(
    centre: tuple[float, float],
    covariance_2d: np.ndarray,
    clr: str,
    label: str | None = None,
    n_sigma: float = 1.0,
) -> Ellipse | None:
    """
    Construct a matplotlib ellipse from a 2-D covariance matrix.

    Parameters
    ----------
    centre:
        Ellipse centre coordinates.
    covariance_2d:
        Two-dimensional covariance matrix defining the ellipse shape and orientation.
    clr:
        Ellipse edge colour.
    label:
        Optional label for the ellipse.
    n_sigma:
        Number of standard deviations represented by the ellipse radius.

    Returns
    -------
    ellipse:
        Ellipse patch representing the covariance, or None if the covariance is not
        finite.

    """

    if not np.all(np.isfinite(covariance_2d)):
        return None

    eigvals, eigvecs = np.linalg.eigh(covariance_2d)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width, height = 2.0 * n_sigma * np.sqrt(np.clip(eigvals, 0.0, np.inf))
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return Ellipse(
        centre,
        width,
        height,
        angle=angle,
        lw=2,
        edgecolor=clr,
        fill=False,
        label=label,
    )

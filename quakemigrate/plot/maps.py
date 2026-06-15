"""
Module for building and plotting map and cross-section visualisations based on
traveltime lookup table geometry.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from quakemigrate.io.station import Station
    from quakemigrate.lut import LUT


@dataclass
class MapAxes2D:
    """Named axes container for a 2-D map panel and its colourbar axes."""

    xy: Axes
    cax: Axes
    bounds: np.ndarray

    def items(self):
        yield "xy", self.xy, (0, 1), ("longitude", "latitude")


def build_2d_map_axes(
    fig: Figure, gs: tuple[int, int], lut: LUT, c: str = "k"
) -> MapAxes2D:
    """
    Build a 2-D map-style axes layout based on the LUT grid dimensions.

    Parameters
    ----------
    fig:
        Figure on which the axes are created.
    gs:
        Grid specification as (nrows, ncols).
    lut:
        Lookup table object providing the map geometry and bounds used for the map
        panel.
    c:
        Colour used for the scalebar.

    Returns
    -------
    axes:
        Dataclass containing the 2-D map axes, colourbar axes, and bounds.

    """

    xy = plt.subplot2grid(gs, (2, 0), colspan=7, rowspan=6, fig=fig)
    cax = plt.subplot2grid(gs, (8, 5), colspan=2, rowspan=1, fig=fig)
    cax.set_axis_off()

    cells_extent = lut.get_grid_extent(cells=True)
    extent = abs(cells_extent[1] - cells_extent[0])
    grid_size = lut.node_spacing * lut.node_count
    aspect = (extent[0] * grid_size[1]) / (extent[1] * grid_size[0])
    xy.set_aspect(aspect=aspect)

    bounds = np.stack(cells_extent, axis=-1)
    gminx, gmaxx = bounds[0]
    gminy, gmaxy = bounds[1]
    xy.set_xlim([gminx, gmaxx])
    xy.set_ylim([gminy, gmaxy])

    num_cells = np.ceil(lut.node_count[0] / 10)
    length = num_cells * lut.node_spacing[0]
    size = extent[0] * length / grid_size[0]
    scalebar = AnchoredSizeBar(
        xy.transData,
        size=size,
        label=f"{length:.3g} {lut.unit_name}",
        loc="lower right",
        pad=0.5,
        sep=5,
        frameon=False,
        color=c,
    )
    xy.add_artist(scalebar)

    xy.tick_params(
        which="both",
        left=True,
        right=True,
        top=True,
        bottom=True,
        labelleft=True,
        labeltop=False,
        labelright=False,
        labelbottom=True,
    )
    xy.set_ylabel("Latitude (deg)", fontsize=14)
    xy.yaxis.set_label_position("left")
    xy.set_xlabel("Longitude (deg)", fontsize=14)

    return MapAxes2D(xy=xy, cax=cax, bounds=bounds)


@dataclass
class MapAxes3D:
    """Named axes container for map/cross-section panels and their colourbar axes."""

    xy: Axes
    xz: Axes
    yz: Axes
    cax: Axes
    bounds: np.ndarray

    def items(self):
        yield "xy", self.xy, (0, 1), ("longitude", "latitude")
        yield "xz", self.xz, (0, 2), ("longitude", "depth")
        yield "yz", self.yz, (2, 1), ("depth", "latitude")


def build_3d_map_axes(
    fig: Figure, gs: tuple[int, int], lut: LUT, c: str = "k"
) -> MapAxes3D:
    """
    Build a 3-D map-style axes layout based on the LUT grid dimensions.

    Parameters
    ----------
    fig:
        Figure on which the axes are created.
    gs:
        Grid specification as (nrows, ncols).
    lut:
        Lookup table object providing the map geometry and bounds used for the map and
        cross-section panels.
    c:
        Colour used for the scalebar.

    Returns
    -------
    axes:
        Dataclass containing the 3-D map axes, colourbar axes, and bounds.

    """

    xy = plt.subplot2grid(gs, (2, 0), colspan=5, rowspan=5, fig=fig)
    xz = plt.subplot2grid(gs, (7, 0), colspan=5, rowspan=2, fig=fig)
    yz = plt.subplot2grid(gs, (2, 5), colspan=2, rowspan=5, fig=fig)
    cax = plt.subplot2grid(gs, (7, 5), colspan=2, rowspan=2, fig=fig)
    cax.set_axis_off()

    xz.sharex(xy)
    yz.sharey(xy)

    cells_extent = lut.get_grid_extent(cells=True)
    extent = abs(cells_extent[1] - cells_extent[0])
    grid_size = lut.node_spacing * lut.node_count
    aspect = (extent[0] * grid_size[1]) / (extent[1] * grid_size[0])
    xy.set_aspect(aspect=aspect)

    bounds = np.stack(cells_extent, axis=-1)
    for i, j, ax in [(0, 1, xy), (0, 2, xz), (2, 1, yz)]:
        gminx, gmaxx = bounds[i]
        gminy, gmaxy = bounds[j]
        ax.set_xlim([gminx, gmaxx])
        ax.set_ylim([gminy, gmaxy])

    num_cells = np.ceil(lut.node_count[0] / 10)
    length = num_cells * lut.node_spacing[0]
    size = extent[0] * length / grid_size[0]
    scalebar = AnchoredSizeBar(
        xy.transData,
        size=size,
        label=f"{length:.3g} {lut.unit_name}",
        loc="lower right",
        pad=0.5,
        sep=5,
        frameon=False,
        color=c,
    )
    xy.add_artist(scalebar)

    xy.tick_params(
        which="both",
        left=True,
        right=True,
        top=True,
        bottom=True,
        labelleft=True,
        labeltop=True,
        labelright=False,
        labelbottom=False,
    )
    xy.set_ylabel("Latitude (deg)", fontsize=14)
    xy.yaxis.set_label_position("left")

    xz.invert_yaxis()
    xz.tick_params(
        which="both",
        left=True,
        right=True,
        top=True,
        bottom=True,
        labelleft=True,
        labeltop=False,
        labelright=False,
        labelbottom=True,
    )
    xz.set_xlabel("Longitude (deg)", fontsize=14)
    xz.set_ylabel(f"Depth ({lut.unit_name})", fontsize=14)
    xz.yaxis.set_label_position("left")

    yz.tick_params(
        which="both",
        left=True,
        right=True,
        top=True,
        bottom=True,
        labelleft=False,
        labeltop=True,
        labelright=True,
        labelbottom=True,
    )
    yz.set_xlabel(f"Depth ({lut.unit_name})", fontsize=14)
    yz.xaxis.set_label_position("bottom")

    return MapAxes3D(xy=xy, xz=xz, yz=yz, cax=cax, bounds=bounds)


def adjust_map_cross_sections(fig: Figure, axes: MapAxes3D) -> None:
    """
    Adjust the cross-section panel dimensions and positions to match the final map-panel
    aspect ratio.

    Parameters
    ----------
    fig:
        Figure containing the map and cross-section axes.
    axes:
        Named map and cross-section axes to be adjusted.

    """

    # Get left, bottom, width, height of each subplot bounding box
    xy_left, xy_bottom, xy_width, xy_height = axes.xy.get_position().bounds
    xz_l, xz_b, xz_w, xz_h = axes.xz.get_position().bounds
    yz_l, yz_b, _, _ = axes.yz.get_position().bounds
    # Find height and width spacing of subplots in figure coordinates
    hdiff = yz_b - (xz_b + xz_h)
    wdiff = yz_l - (xz_l + xz_w)
    # Adjust bottom of xz cross section (if bottom of map has moved up)
    new_xz_bottom = xy_bottom - hdiff - xz_h
    axes.xz.set_position([xy_left, new_xz_bottom, xy_width, xz_h])
    # Adjust left of yz cross section (if right side of map has moved left)
    new_yz_left = xy_left + xy_width + wdiff
    # Take this opportunity to ensure the height of both cross sections is
    # equal by adjusting yz width (almost there from gridspec maths already)
    new_yz_width = xz_h * (fig.get_size_inches()[1] / fig.get_size_inches()[0])
    axes.yz.set_position([new_yz_left, xy_bottom, new_yz_width, xy_height])


def plot_stations(axes: MapAxes2D | MapAxes3D, stations: list[Station], c: str) -> None:
    """
    Plot station markers on map and cross-section panels.

    Station names are annotated on the XY panel only.

    Parameters
    ----------
    axes:
        Map and optional cross-section axes on which to plot stations.
    stations:
        List of Station objects to be visualised on the map and cross-sections.
    c:
        Marker and annotation colour.

    """

    for ax_label, ax, _, (x, y) in axes.items():
        marker = "<" if ax_label == "yz" else "^"
        ax.scatter(
            [getattr(station, x) for station in stations],
            [getattr(station, y) for station in stations],
            s=15,
            marker=marker,
            zorder=20,
            c=c,
        )

        if ax_label == "xy":
            for station in stations:
                ax.annotate(
                    station.id,
                    [station.longitude, station.latitude],
                    zorder=20,
                    c=c,
                    clip_on=True,
                )


@dataclass
class XYOverlaySpec:
    """Specification for a single user-supplied map overlay."""

    file: pathlib.Path
    kind: Literal["line", "scatter"] = "line"
    color: str = "black"
    linewidth: float = 1.0
    linestyle: str = "-"
    marker: str = "o"
    markersize: float = 20.0
    alpha: float = 1.0
    label: str | None = None


def _read_xy_overlay_manifest(xy_files: str | pathlib.Path) -> list[XYOverlaySpec]:
    """
    Read a map-overlay manifest file into structured overlay specifications.

    The manifest is a CSV file with one overlay per row. Only the first two columns are
    required; trailing style columns may be omitted and are replaced with defaults. The
    columns are:

    File, Kind, Color, Size, Style, Alpha, Label

    The meaning of Size and Style depends on Kind:

    - for line overlays, Size is linewidth and Style is linestyle
    - for scatter overlays, Size is markersize and Style is marker

    Parameters
    ----------
    xy_files:
        Path to a CSV manifest describing one or more map overlays.

    Returns
    -------
    specs:
        List of parsed overlay specifications.

    Raises
    ------
    ValueError
        Raised if a row is empty, an overlay file is missing, an overlay kind is not
        recognised, or a numeric style field cannot be parsed.

    """

    manifest = pathlib.Path(xy_files)

    specs = []
    with manifest.open(newline="") as f:
        reader = csv.reader(f)

        for line_number, row in enumerate(reader, start=1):
            row = [value.strip() for value in row]

            if not row or not any(row):
                continue

            if row[0].startswith("#"):
                continue

            specs.append(_parse_xy_overlay_row(row, manifest, line_number))

    return specs


def _parse_xy_overlay_row(
    row: list[str],
    manifest: pathlib.Path,
    line_number: int,
) -> XYOverlaySpec:
    """
    Parse a single compact overlay manifest row.

    Parameters
    ----------
    row:
        CSV row values with surrounding whitespace stripped.
    manifest:
        Path to the manifest file, used for relative overlay paths and error messages.
    line_number:
        Row number in the manifest file, counted from 1.

    Returns
    -------
    spec:
        Parsed overlay specification.

    Raises
    ------
    ValueError
        Raised if the row is invalid.

    """

    row += [""] * (7 - len(row))
    file_, kind, color, size, style, alpha, label = row[:7]

    if not file_:
        raise ValueError(f"Missing overlay file in {manifest} on line {line_number}.")

    kind = kind or "line"
    if kind not in {"line", "scatter"}:
        raise ValueError(
            f"Invalid overlay kind {kind} in {manifest} on line {line_number}. "
            "Expected 'line' or 'scatter'."
        )

    file_path = pathlib.Path(file_)
    if not file_path.is_absolute():
        file_path = manifest.parent / file_path

    try:
        if kind == "line":
            return XYOverlaySpec(
                file=file_path,
                kind="line",
                color=color or "black",
                linewidth=float(size) if size else 1.0,
                linestyle=style or "-",
                alpha=float(alpha) if alpha else 1.0,
                label=label or None,
            )

        return XYOverlaySpec(
            file=file_path,
            kind="scatter",
            color=color or "black",
            markersize=float(size) if size else 20.0,
            marker=style or "o",
            alpha=float(alpha) if alpha else 1.0,
            label=label or None,
        )

    except ValueError as e:
        raise ValueError(
            f"Invalid numeric overlay style value in {manifest} on line "
            f"{line_number}: {row[:7]}"
        ) from e


def _read_xy_points(file: str | pathlib.Path) -> pd.DataFrame:
    """
    Read a two-column longitude/latitude coordinate file.

    Parameters
    ----------
    file:
        Path to a coordinate file containing longitude and latitude pairs.

    Returns
    -------
    points:
        DataFrame containing Longitude and Latitude columns.

    Raises
    ------
    ValueError
        If the file contains fewer than two columns.

    """

    df = pd.read_csv(
        file,
        names=["Longitude", "Latitude"],
        header=None,
        comment="#",
    )

    if df.shape[1] < 2:
        raise ValueError(
            f"{file} must contain at least two columns: Longitude, Latitude"
        )

    return df[["Longitude", "Latitude"]]


def plot_map_overlays(overlay_manifest: str | pathlib.Path, ax: Axes) -> None:
    """
    Plot user-supplied overlays on a map axes.

    The overlay manifest file contains one row per overlay and may specify the overlay
    kind and plotting style. Supported overlay kinds are line and scatter.

    The manifest columns are interpreted as:

    File, Kind, Color, Linewidth, Linestyle, Marker, Markersize, Alpha, Label

    Each referenced coordinate file should contain longitude and latitude pairs,
    one point per row. Lines beginning with # are treated as comments.

    Parameters
    ----------
    overlay_manifest:
        Path to a CSV overlay manifest file.
    ax:
        Axes on which to plot the overlays.

    """

    specs = _read_xy_overlay_manifest(overlay_manifest)

    for spec in specs:
        xy = _read_xy_points(spec.file)

        x, y = xy["Longitude"].values, xy["Latitude"].values

        if spec.kind == "line":
            ax.plot(
                x,
                y,
                color=spec.color,
                linewidth=spec.linewidth,
                linestyle=spec.linestyle,
                alpha=spec.alpha,
                label=spec.label,
                zorder=40,
            )
        elif spec.kind == "scatter":
            ax.scatter(
                x,
                y,
                c=spec.color,
                s=spec.markersize,
                marker=spec.marker,
                alpha=spec.alpha,
                label=spec.label,
                zorder=40,
            )

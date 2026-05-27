"""
Plugins for located event summaries.

This module provides visualisation plugins that generate standard QuakeMigrate event
summary figures after an event has been located.

The plugins are intended to be constructed from plugin configuration and executed during
the locate stage. They wrap the existing plotting functions in quakemigrate.plot.event
so that event summary generation can be enabled, disabled, ordered, and configured using
the plugin system.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from quakemigrate.plot.event import event_summary_2d, event_summary_3d


if TYPE_CHECKING:
    import numpy as np

    from quakemigrate.io.core import Run
    from quakemigrate.io.event import Event
    from quakemigrate.lut import LUT


@dataclass
class EventSummary3DPlugin:
    """
    Plugin that generates a 3-D event summary plot for a located event.

    This plugin runs during the locate stage and calls
    :func:`quakemigrate.plot.event.event_summary_3d` using the located event, lookup
    table, run metadata, and marginalised coalescence map supplied in the runtime
    context.

    Attributes
    ----------
    stage:
        Processing stage in which the plugin is executed.
    order:
        Relative execution order within the stage. Lower values run earlier.
    name:
        Plugin name used in configuration and reporting.
    kind:
        Plugin category.
    overlay_manifest:
        Optional path to an overlay manifest used to add overlays to the plot.
    plot_all_stations:
        Whether to plot all stations, rather than only stations used for the event.
    file_type:
        Output file type passed to the plotting function.

    """

    stage: str = "locate_event"
    order: int = 450
    name: str = "EventSummary3D"
    kind: str = "visualisation"

    overlay_manifest: str | None = None
    plot_all_stations: bool = True
    file_type: str = "pdf"

    def run(
        self,
        event: Event,
        lut: LUT,
        run: Run,
        marginalised_coa_map: np.ndarray,
    ) -> None:
        """
        Generate the 3-D event summary plot.

        Parameters
        ----------
        event:
            Located event to summarise.
        lut:
            Lookup table used for event location.
        run:
            QuakeMigrate run metadata and output configuration.
        marginalised_coa_map:
            Marginalised coalescence map for the located event.

        """

        event_summary_3d(
            run,
            event,
            marginalised_coa_map,
            lut,
            overlay_manifest=self.overlay_manifest,
            plot_all_stations=self.plot_all_stations,
            file_type=self.file_type,
        )


@dataclass
class EventSummary2DPlugin:
    """
    Plugin that generates a 2-D event summary plot for a located event.

    This plugin runs during the locate stage and calls
    :func:`quakemigrate.plot.event.event_summary_2d` using the located event, lookup
    table, run metadata, and marginalised coalescence map supplied in the runtime
    context.

    Attributes
    ----------
    stage:
        Processing stage in which the plugin is executed.
    order:
        Relative execution order within the stage. Lower values run earlier.
    name:
        Plugin name used in configuration and reporting.
    kind:
        Plugin category.
    overlay_manifest:
        Optional path to an overlay manifest used to add overlays to the plot.
    slice_mode:
        Type of 2-D slice to plot. "maximum" plots the maximum-amplitude slice;
        "surface" plots a fixed-depth surface slice.
    surface_depth:
        Depth of the surface slice when slice_mode is "surface".
    plot_all_stations:
        Whether to plot all stations, rather than only stations used for the event.
    file_type:
        Output file type passed to the plotting function.

    """

    stage: str = "locate_event"
    order: int = 451
    name: str = "EventSummary2D"
    kind: str = "visualisation"

    overlay_manifest: str | None = None
    slice_mode: Literal["surface", "maximum"] = "maximum"
    surface_depth: float = 0.0
    plot_all_stations: bool = True
    file_type: str = "pdf"

    def run(
        self,
        event: Event,
        lut: LUT,
        run: Run,
        marginalised_coa_map: np.ndarray,
    ) -> None:
        """
        Generate the 2-D event summary plot.

        Parameters
        ----------
        event:
            Located event to summarise.
        lut:
            Lookup table used for event location.
        run:
            QuakeMigrate run metadata and output configuration.
        marginalised_coa_map:
            Marginalised coalescence map for the located event.

        """

        event_summary_2d(
            run,
            event,
            marginalised_coa_map,
            lut,
            slice_mode=self.slice_mode,
            surface_depth=self.surface_depth,
            overlay_manifest=self.overlay_manifest,
            plot_all_stations=self.plot_all_stations,
            file_type=self.file_type,
        )

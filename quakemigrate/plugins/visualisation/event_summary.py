"""
Plugins for located event summaries.

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

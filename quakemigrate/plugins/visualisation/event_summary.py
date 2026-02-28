"""
Plugins for located event summaries.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

from quakemigrate.plot.event import event_summary_2d, event_summary_3d


@dataclass
class EventSummary3DPlugin:
    stage: str = "locate_event"
    order: int = 450
    name: str = "event_summary_3d"

    enabled_flag: bool = True
    xy_files: str | None = None
    plot_all_stns: bool = True
    file_type: str = "pdf"

    def enabled(self, **_: Any) -> bool:
        return self.enabled_flag

    def run(
        self,
        event,
        lut,
        run,
        marginalised_coa_map,
        **_: Any,
    ) -> Mapping[str, Any] | None:
        event_summary(
            run,
            event,
            marginalised_coa_map,
            lut,
            xy_files=self.xy_files,
            plot_all_stns=self.plot_all_stns,
            file_type=self.file_type,
        )

        outdir: pathlib.Path = run.path / "locate" / run.subname / "summaries"
        fstem = f"{run.name}_{event.uid}_EventSummary"
        outpath = (outdir / fstem).with_suffix(f".{self.file_type}")

        return {"event_summary_path": outpath}


@dataclass
class EventSummary2DPlugin:
    stage: str = "locate_event"
    order: int = 450
    name: str = "event_summary_2d"

    enabled_flag: bool = True
    xy_files: str | None = None
    plot_all_stns: bool = True
    file_type: str = "pdf"

    def enabled(self, **_: Any) -> bool:
        return self.enabled_flag

    def run(
        self,
        event,
        lut,
        run,
        marginalised_coa_map,
        **_: Any,
    ) -> Mapping[str, Any] | None:
        event_summary_2d(
            run,
            event,
            marginalised_coa_map,
            lut,
            xy_files=self.xy_files,
            plot_all_stns=self.plot_all_stns,
            file_type=self.file_type,
        )

        outdir: pathlib.Path = run.path / "locate" / run.subname / "summaries"
        fstem = f"{run.name}_{event.uid}_EventSummary"
        outpath = (outdir / fstem).with_suffix(f".{self.file_type}")

        return {"event_summary_path": outpath}

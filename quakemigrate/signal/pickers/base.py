"""
A simple abstract base class with method stubs enabling simple modification of
QuakeMigrate to use custom phase picking methods that remain compatible with the core of
the package.

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

import quakemigrate


class PhasePicker(ABC):
    """
    Abstract base class providing a simple way of modifying the default picking function
    in QuakeMigrate.

    Attributes
    ----------
    plot_picks : bool
        Toggle plotting of phase picks.

    """

    def __init__(self, **kwargs) -> None:
        """Instantiate the PhasePicker object."""
        self.plot_picks = kwargs.get("plot_picks", True)

    def __str__(self) -> str:
        """Returns a short summary string of the PhasePicker object."""
        return (
            "Abstract PhasePicker object - consider adding a __repr__ "
            "method to your custom PhasePicker class that gives the user "
            "relevant information about the object."
        )

    @abstractmethod
    def pick_phases(self):
        """Method stub for phase picking."""
        pass

    def write(
        self, run: quakemigrate.io.core.Run, event_uid: str, phase_picks: pd.DataFrame
    ) -> None:
        """
        Write phase picks to a new .picks file.

        Parameters
        ----------
        run:
            Light class encapsulating i/o path information for a given run.
        event_uid:
            Unique identifier for the event.
        phase_picks:
            Phase pick times with columns: ["Name", "Phase", "ModelledTime",
            "PickTime", "PickError", "SNR"]
            Each row contains the phase pick from one station/phase.

        """

        fpath = run.path / "locate" / run.subname / "picks"
        fpath.mkdir(exist_ok=True, parents=True)

        # Work on a copy
        phase_picks = phase_picks.copy()

        # Set floating point precision for output file
        for col in ["PickError", "SNR"]:
            phase_picks[col] = phase_picks[col].map(
                lambda x: f"{x:.3g}", na_action="ignore"
            )

        fstem = f"{event_uid}"
        fname = (fpath / fstem).with_suffix(".picks")
        phase_picks.to_csv(fname, index=False)

    def plot(self) -> None:
        """Method stub for phase pick plotting."""
        print(
            "Consider adding a plot method to your custom PhasePicker"
            " class - see the GaussianPicker class for reference."
        )

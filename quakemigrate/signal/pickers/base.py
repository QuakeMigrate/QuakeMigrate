# -*- coding: utf-8 -*-
"""
A simple abstract base class with method stubs enabling simple modification of
QuakeMigrate to use custom phase picking methods that remain compatible with the core of
the package.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from abc import ABC, abstractmethod


class PhasePicker(ABC):
    """
    Abstract base class providing a simple way of modifying the default picking function
    in QuakeMigrate.

    Attributes
    ----------
    plot_picks : bool
        Toggle plotting of phase picks.

    Methods
    -------
    pick_phases
        Abstract method stub providing interface with QuakeMigrate scan.
    write(event_uid, phase_picks, output)
        Outputs phase picks to file.
    plot
        Method stub for phase pick plotting.

    """

    def __init__(self, **kwargs):
        """Instantiate the PhasePicker object."""
        self.plot_picks = kwargs.get("plot_picks", True)

    def __str__(self):
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

    def write(self, run, event_uid, phase_picks):
        """
        Write phase picks to a new .picks file.

        Parameters
        ----------
        event_uid : str
            Unique identifier for the event.
        phase_picks : pandas DataFrame object
            Phase pick times with columns: ["Name", "Phase",
                                            "ModelledTime",
                                            "PickTime", "PickError",
                                            "SNR"]
            Each row contains the phase pick from one station/phase.
        output : QuakeMigrate input/output control object
            Contains useful methods controlling output for the scan.

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

    def plot(self):
        """Method stub for phase pick plotting."""
        print(
            "Consider adding a plot method to your custom PhasePicker"
            " class - see the GaussianPicker class for reference."
        )

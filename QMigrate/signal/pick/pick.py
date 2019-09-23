# -*- coding: utf-8 -*-
"""
A simple abstract base class with method stubs to enable users to extend
QuakeMigrate with custom phase picking methods that remain compatible with the
core of the package.

"""

from abc import ABC, abstractmethod


class PhasePicker(ABC):
    """
    QuakeMigrate default pick function class.

    Attributes
    ----------
    plot_phase_picks : bool
        Toggle plotting of phase picks

    Methods
    -------
    pick_phases()

    """

    def __init__(self):
        """Class initialisation method."""

        self.plot_phase_picks = False

    def __str__(self):
        """
        Return short summary string of the Pick object

        It should provide information on all of the various parameters that the
        user can/has set.

        """

        out = "Default Pick object - add a __str__ method to your Pick class"

        return out

    @abstractmethod
    def pick_phases(self):
        """Method stub for phase picking"""
        pass

    @abstractmethod
    def plot(self):
        """Method stub for plotting of phase picks"""
        pass

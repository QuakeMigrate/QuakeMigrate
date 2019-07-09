# -*- coding: utf-8 -*-
"""
A simple abstract base class that offers access points for users to extend
QuakeMigrate with custom onset functions that is compatible with the core of
the package.

"""

from abc import ABC, abstractmethod


class Onset(ABC):
    """
    QuakeMigrate default onset function class.

    Methods
    -------
    p_onset()
        Generate an onset function that represents the P-phase arrival

    s_onset()
        Generate an onset function that represents the S-phase arrival

    """

    def __init__(self):
        """Class initialisation method."""

        # Default data sampling rate
        self.sampling_rate = 50

        # Default pre-pad/post-pad
        self._pre_pad = 0
        self._post_pad = 0

    @abstractmethod
    def __str__(self):
        """
        Return short summary string of the Onset object

        It will provide information on all of the various parameters that the
        user can/has set.

        """

        out = "Default Onset object - add a __str__ method to your Onset class"

        return out

    @abstractmethod
    def p_onset(self):
        """Method stub for p_onset."""
        pass

    @abstractmethod
    def s_onset(self):
        """Method stub for s_onset."""
        pass

    @property
    @abstractmethod
    def pre_pad(self):
        """Get property stub for pre_pad."""
        return self._pre_pad

    @pre_pad.setter
    @abstractmethod
    def pre_pad(self, value):
        """Set property stub for pre_pad."""
        self._pre_pad = value

    @property
    @abstractmethod
    def post_pad(self):
        """Get property stub for pre_pad."""
        return self._post_pad

    @post_pad.setter
    @abstractmethod
    def post_pad(self, value):
        """Set property stub for pre_pad."""
        self._post_pad = value

    @property
    def signal(self):
        """Get signal"""

        return self._signal

    @signal.setter
    def signal(self, signal):
        """Set signal"""

        self._signal = signal
        self.sige = signal[0]
        self.sign = signal[1]
        self.sigz = signal[2]

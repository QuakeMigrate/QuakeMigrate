# -*- coding: utf-8 -*-
"""
A simple abstract base class with method stubs to enable users to extend
QuakeMigrate with custom onset functions that remain compatible with the core
of the package.

"""

from abc import ABC, abstractmethod

import numpy as np

import quakemigrate.util as util


class Onset(ABC):
    """
    QuakeMigrate default onset function class.

    Attributes
    ----------
    sampling_rate : int
        Desired sampling rate for input data; sampling rate at which the onset
        functions will be computed.
    pre_pad : float, optional
        Option to override the default pre-pad duration of data to read before
        computing 4-D coalescence in detect() and locate().
    post_pad : float
        Option to override the default post-pad duration of data to read before
        computing 4-D coalescence in detect() and locate().

    Methods
    -------
    calculate_onsets()
        Generate onset functions that represent seismic phase arrivals

    """

    def __init__(self, **kwargs):
        """Instantiate the Onset object."""

        self.sampling_rate = kwargs.get("sampling_rate", 50)
        self._pre_pad = 0
        self._post_pad = 0

    def __str__(self):
        """Return short summary string of the Onset object."""

        return "Base Onset object - add a __str__ method to your Onset class"

    def pad(self, timespan):
        """
        Determine the number of samples needed to pre- and post-pad the
        timespan.

        Parameters
        ----------
        timespan : float
            The time window to pad.

        Returns
        -------
        pre_pad : float
            Option to override the default pre-pad duration of data to read
            before computing 4-D coalescence in detect() and locate().
        post_pad : float
            Option to override the default post-pad duration of data to read
            before computing 4-D coalescence in detect() and locate().

        """

        # Add additional padding for any tapering applied to data
        timespan += (self.pre_pad + self.post_pad)
        pre_pad = util.trim2sample(self.pre_pad + np.ceil(timespan*0.06),
                                   self.sampling_rate)
        post_pad = util.trim2sample(self.post_pad + np.ceil(timespan*0.06),
                                    self.sampling_rate)

        return pre_pad, post_pad

    def gaussian_halfwidth(self, phase):
        """Method stub for Gaussian half-width estimate."""

        msg = ("In order to use the GaussianPicker module with a custom Onset,"
               " you need to provide a 'gaussian_halfwidth' method.")

        raise AttributeError(msg)

    @abstractmethod
    def calculate_onsets(self):
        """Method stub for calculation of onset functions."""
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

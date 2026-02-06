"""
Module to handle input/output for QuakeMigrate.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
import pathlib
import pickle
from typing import Literal, TYPE_CHECKING

import pandas as pd
from obspy import read_inventory

import quakemigrate.util as util
from quakemigrate.lut import LUT


if TYPE_CHECKING:
    from obspy.core.inventory import Inventory


def read_lut(lut_file: str) -> LUT:
    """
    Read the contents of a pickle file and restore state of the lookup table object.

    Parameters
    ----------
    lut_file:
        Path to pickle file to load.

    Returns
    -------
    lut:
        Lookup table populated with grid specification and traveltimes.

    """

    lut = LUT()
    with open(lut_file, "rb") as f:
        lut.__dict__.update(pickle.load(f))

    if hasattr(lut, "maps"):
        print(
            "FutureWarning: The internal data structure of LUT has changed."
            "\nTo remove this warning you will need to convert your lookup "
            "table to the new-style\nusing `quakemigrate.lut.update_lut`."
        )

    return lut


def stations(station_file: str, **kwargs: dict) -> pd.DataFrame:
    """Alias for read_stations."""
    print(
        "FutureWarning: function name has changed - continuing.\n"
        "To remove this message, change:\t'stations' -> 'read_stations'"
    )

    return read_stations(station_file, **kwargs)


def read_stations(station_file: str, **kwargs: dict) -> pd.DataFrame:
    """
    Reads station information from file.

    Parameters
    ----------
    station_file:
        Path to station file.
        File format (header line is REQUIRED, case sensitive, any order):
            Latitude, Longitude, Elevation (units matching LUT grid projection;
            either metres or kilometres; positive upwards), Name
    kwargs:
        Passthrough for `pandas.read_csv` kwargs.

    Returns
    -------
    stn_data:
        Columns:
            "Latitude", "Longitude", "Elevation", "Name"

    Raises
    ------
    StationFileHeaderException
        Raised if the input file is missing required entries in the header.

    """

    stn_data = pd.read_csv(station_file, **kwargs)

    if ("Latitude" or "Longitude" or "Elevation" or "Name") not in stn_data.columns:
        raise util.StationFileHeaderException

    stn_data["Elevation"] = stn_data["Elevation"].apply(lambda x: -1 * x)

    # Ensure station names are strings
    stn_data = stn_data.astype({"Name": "str"})

    return stn_data


def read_response_inv(
    response_file: str,
    ignore_network_code: bool = False,
    dummy_network_code: str = "XX",
    ignore_location_code: bool = False,
    sac_pz_format: bool = False,
) -> Inventory:
    """
    Reads response information from file, returning it as a `obspy.Inventory` object.

    Parameters
    ----------
    response_file:
        Path to response file.
        Please see the `obspy.read_inventory()` documentation for a full list of
        supported file formats. This includes a dataless.seed volume, a concatenated
        series of RESP files or a stationXML file.
    ignore_network_code:
        If True, replace all network codes in the waveform archive with a dummy value.
        Note this may cause issues if station codes are repeated, with SEED-ID's only
        distinguished by their differing network codes.
    dummy_network_code:
        Provides the option to specify the dummy network code applied to the waveform
        archive, if `ignore_network_code` is set to True.
    ignore_location_code:
        If True, replace all location codes in the waveform archive with a blank string.
    sac_pz_format:
        Toggle to indicate that response information is being provided in SAC Pole-Zero
        files. NOTE: not yet supported.

    Returns
    -------
    response_inv:
        ObsPy response inventory.

    Raises
    ------
    NotImplementedError
        If the user selects `sac_pz_format=True`.
    TypeError
        If the user provides a response file that is not readable by ObsPy.

    """

    if sac_pz_format:
        raise NotImplementedError(
            "SAC_PZ is not yet supported. Please contact the QuakeMigrate developers."
        )
    else:
        try:
            response_inv = read_inventory(response_file)
        except TypeError as e:
            raise TypeError(
                f"Response file not readable by ObsPy: {e}\n"
                "Please consult the ObsPy documentation."
            )

        if ignore_network_code:
            if isinstance(dummy_network_code, str) and len(dummy_network_code) == 2:
                for network in response_inv.networks:
                    network.code = dummy_network_code
            else:
                raise ValueError(
                    f"dummy_network_code must be a 2 character string, not {dummy_network_code}."
                )

        if ignore_location_code:
            for network in response_inv.networks:
                for station in network.stations:
                    for chan in station.channels:
                        chan.location_code = ""

    return response_inv


def read_vmodel(vmodel_file: str, **kwargs: dict) -> pd.DataFrame:
    """
    Reads velocity model information from file.

    Parameters
    ----------
    vmodel_file:
        Path to velocity model file.
        File format: (header line is REQUIRED, case sensitive, any order):
            "Depth" of each layer in the model (units matching the LUT grid
            projection; positive-down)
            "V<phase>" velocity for each layer in the model, for each phase
            the user wishes to calculate traveltimes for (units matching the
            LUT grid projection). There are no required phases, and no maximum
            number of separate phases, e.g., "Vp", "Vs", "Vsh".
    kwargs:
        Passthrough for `pandas.read_csv` kwargs.

    Returns
    -------
    vmodel_data:
        Columns:
            "Depth" of each layer in model (positive down)
            "V<phase>" velocity for each layer in model (e.g., "Vp")

    Raises
    ------
    VelocityModelFileHeaderException
        Raised if the input file is missing required entries in the header.

    """

    vmodel_data = pd.read_csv(vmodel_file, **kwargs)

    if "Depth" not in vmodel_data.columns:
        raise util.InvalidVelocityModelHeader("Depth")

    return vmodel_data


class Run:
    """
    Light class to encapsulate i/o path information for a given run.

    Parameters
    ----------
    path:
        Points to the top level directory containing all input files, under which the
        specific run directory will be created.
    name:
        Name of the current QuakeMigrate run.
    subname:
        Optional name of a sub-run - useful when testing different trigger parameters,
        for example.
    stage:
        Specifies run stage of QuakeMigrate.
    loglevel:
        Set the logging level. (Default "info")

    Attributes
    ----------
    name:
        Name of the current QuakeMigrate run.

    """

    def __init__(
        self,
        path: pathlib.Path,
        name: str,
        subname: str | None = None,
        stage: Literal["detect", "trigger", "locate"] | None = None,
        loglevel: str = "info",
    ) -> None:
        """Instantiate the Run object."""

        if subname is None:
            subname = ""

        if "." in name or "." in subname:
            print(
                "Warning: The character '.' is not allowed in run names/subnames - "
                "replacing with '_'."
            )
            name = name.replace(".", "_")
            subname = subname.replace(".", "_")

        self.path = pathlib.Path(path) / name
        self._name = name
        self.stage = stage
        self.subname = subname
        self.loglevel = loglevel

    def __str__(self) -> str:
        """Return short summary string of the Run object."""

        return (
            f"{util.log_spacer}\n{util.log_spacer}\n"
            f"\tQuakeMigrate RUN - Path: {self.path} - Name: {self.name}\n"
            f"{util.log_spacer}\n{util.log_spacer}\n"
        )

    def logger(self, log: bool) -> None:
        """
        Configures the logging feature.

        Parameters
        ----------
        log:
            Toggle for logging. If True, will output to stdout and generate a log file.

        """

        logstem = self.path / self.stage / self.subname / "logs" / self.name
        util.logger(logstem, log, loglevel=self.loglevel)
        logging.info(self)

    @property
    def name(self) -> str:
        """Get the run name as a formatted string."""
        if self.subname == "":
            return self._name
        else:
            return f"{self._name}_{self.subname}"

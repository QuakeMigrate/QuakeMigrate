"""
Waveform client implementations and factory helpers.

This package provides waveform clients for reading miniSEED and other ObsPy-readable
waveform data from:

- a local filesystem archive with a regular path pattern
- a remote FDSN web service
- a SeisMon backend

It also exposes a factory function, :func:`make_waveform_client`, for constructing a
waveform client instance from a configuration mapping.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from typing import Any, Mapping

from quakemigrate.io import read_response_inv
from .base import BaseWaveformClient
from .fdsn import FDSNWaveformClient
from .local import LocalWaveformClient
from .seismonpy import SeismonWaveformClient


def make_waveform_client(config: Mapping[str, Any]) -> BaseWaveformClient:
    """
    Construct a waveform client from a configuration mapping.

    Parameters
    ----------
    config:
        Configuration specifying the waveform client type and its options.
        The ``client`` key selects the backend implementation. Supported values are
        ``"local"``, ``"fdsn"``, and ``"seismon"``.

        If ``inventory_path`` is provided, the inventory is read with ObsPy and passed
        to the constructed client as ``inventory``.

    Returns
    -------
    client:
        Configured waveform client instance.

    Raises
    ------
    ValueError
        Raised if ``config["client"]`` specifies an unknown client type.
    KeyError
        Raised if required configuration keys for the selected client are missing.

    """

    client_config = dict(config)

    client_type = client_config.pop("client")
    inventory_path = client_config.pop("inventory_path", None)
    inventory = (
        read_response_inv(inventory_path) if inventory_path is not None else None
    )

    match client_type:
        case "local":
            return LocalWaveformClient(
                path=client_config.pop("path"),
                format=client_config.pop("format"),
                inventory=inventory,
                **client_config,
            )
        case "fdsn":
            return FDSNWaveformClient(
                base_url=client_config.pop("base_url"),
                timeout=int(client_config.pop("timeout", 60)),
                inventory=inventory,
                **client_config,
            )
        case "seismon":
            return SeismonWaveformClient(inventory=inventory, **client_config)
        case _:
            raise ValueError(f"Unknown client: {client_type}")


__all__ = [
    "BaseWaveformClient",
    "LocalWaveformClient",
    "FDSNWaveformClient",
    "SeismonWaveformClient",
    "make_waveform_client",
]

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

from quakemigrate.clients.base import BaseWaveformClient
from quakemigrate.clients.registry import get_waveform_client_class
from quakemigrate.io import read_response_inv


def make_waveform_client(config: Mapping[str, Any]) -> BaseWaveformClient:
    """
    Construct a waveform client from a configuration mapping.

    Parameters
    ----------
    config:
        Configuration specifying the waveform client type and its options.
        The client key selects the backend implementation. Supported values are
        "local", "fdsn", and any custom plugin registered to the
        quakemigrate.waveform_clients entrypoint.

        If inventory_path is provided, the inventory is read with ObsPy and passed
        to the constructed client as inventory.

    Returns
    -------
    client:
        Configured waveform client instance.

    Raises
    ------
    ValueError
        Raised if config["client"] specifies an unknown client type.
    KeyError
        Raised if required configuration keys for the selected client are missing.

    """

    client_config = dict(config)

    client_type = client_config.pop("client")
    inventory_path = client_config.pop("inventory_path", None)
    inventory = (
        read_response_inv(inventory_path) if inventory_path is not None else None
    )

    waveform_client_class = get_waveform_client_class(client_type)

    return waveform_client_class(
        inventory=inventory,
        **client_config,
    )


__all__ = [
    "BaseWaveformClient",
    "make_waveform_client",
]

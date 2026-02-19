"""
Backward-compatible imports for moved phase picking code.

Deprecated:
    quakemigrate.signal.pickers

New location:
    quakemigrate.plugins.pickers

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import importlib
import warnings
from typing import Any


__all__ = ["Onset", "OnsetData", "STALTAOnset"]

_MOVED = {
    "PhasePicker": ("quakemigrate.plugins.pickers.base", "PhasePicker"),
    "GaussianPicker": ("quakemigrate.plugins.picker.gaussian", "GaussianPicker"),
}

_warned: set[str] = set()


def __getattr__(name: str) -> Any:
    if name not in _MOVED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    new_mod, new_name = _MOVED[name]

    if name not in _warned:
        warnings.warn(
            f"`{__name__}.{name}` is deprecated and will be removed in a future "
            f"release.\nImport from `{new_mod}` instead:\n"
            f"    from {new_mod} import {new_name}",
            DeprecationWarning,
            stacklevel=2,
        )
        _warned.add(name)

    mod = importlib.import_module(new_mod)
    obj = getattr(mod, new_name)

    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_MOVED.keys()))

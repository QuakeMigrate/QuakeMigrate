"""
Collection of plugins for use with the core QuakeMigrate package.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Plugin(Protocol):
    stage: str
    order: int
    name: str

    def enabled(self, **ctx: Any) -> bool:
        """Return True if plugin should run for this context."""
        ...

    def run(self, **ctx: Any) -> Mapping[str, Any] | None:
        """Execute plugin. May mutate domain objects. May return artifacts to merge into ctx."""
        ...


def call_by_signature(fn: Callable, available: dict[str, Any]):
    """
    Utility function that handles dependency injection based on the signature of the function.

    """

    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in available:
            kwargs[name] = available[name]
        elif param.default is inspect._empty:
            raise TypeError(f"Missing required dependency '{name}' for {fn}")
    return fn(**kwargs)

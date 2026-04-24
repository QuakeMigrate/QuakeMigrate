"""
Compatibility shim.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import warnings

from quakemigrate.plugins.onsets.base import Onset


warnings.warn(
    "quakemigrate.signal.onsets.base is deprecated. "
    "Use quakemigrate.plugins.onsets.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

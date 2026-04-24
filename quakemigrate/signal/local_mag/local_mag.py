"""
Compatibility shim.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import warnings

from quakemigrate.plugins.magnitudes.local_mag import LocalMag


warnings.warn(
    "quakemigrate.signal.local_mag.local_mag is deprecated. "
    "Use quakemigrate.plugins.magnitudes.local_mag instead.",
    DeprecationWarning,
    stacklevel=2,
)

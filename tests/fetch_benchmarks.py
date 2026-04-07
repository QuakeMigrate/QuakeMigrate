"""
Download and extract QuakeMigrate benchmark data from Zenodo.

The benchmark archive is cached using pooch in an OS-appropriate user cache.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pathlib

import pooch


ZENODO_URL = "https://zenodo.org/records/19350669/files/benchmarks.zip?download=1"
ZENODO_HASH = "md5:4128e4922c422f5a84e011f56af22bf3"


def fetch_benchmarks() -> pathlib.Path:
    """
    Download and extract the benchmark archive from Zenodo, checking the MD5 hash.

    Returns
    -------
    extract_dir:
        Path to the extracted benchmark root directory.

    """

    cache_path = pathlib.Path(pooch.os_cache("quakemigrate"))
    extract_dir = pathlib.Path.cwd()

    pooch.retrieve(
        url=ZENODO_URL,
        known_hash=ZENODO_HASH,
        fname="benchmarks.zip",
        path=cache_path,
        processor=pooch.Unzip(extract_dir=extract_dir),
    )

    return extract_dir


def main() -> None:
    benchmark_dir = fetch_benchmarks()
    print(f"Benchmark data available in: {benchmark_dir.resolve()}")


if __name__ == "__main__":
    main()

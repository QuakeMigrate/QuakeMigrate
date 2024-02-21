# -*- coding: utf-8 -*-
"""
This script demonstrates how to run the detect stage of QuakeMigrate.

For more details, please see the manual and read the docs.

"""

# Stop numpy using all available threads (these environment variables must be
# set before numpy is imported for the first time).
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

from quakemigrate import QuakeScan
from quakemigrate.io import Archive, read_lut, read_stations
from quakemigrate.signal.onsets import STALTAOnset

# --- i/o paths ---
archive_path = "/path/to/archived/data"
lut_file = "/path/to/lut_file"
station_file = "/path/to/station_file"

run_path = "/path/to/output"
run_name = "name_of_run"

# --- Set time period over which to run detect ---
starttime = "2018-001T00:00:00.0"
endtime = "2018-002T00:00:00.0"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Create new Archive and set path structure ---
archive = Archive(
    archive_path=archive_path, stations=stations, archive_format="YEAR/JD/STATION"
)
# For custom structures...
# archive.format = "custom/archive_{year}_{jday}/{month:02d}-{day:02d}.{station}_structure"

# --- Resample data with mismatched sampling rates ---
# archive.resample = True
# archive.upfactor = 2

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)

# --- Decimate the lookup table ---
lut = lut.decimate([2, 2, 2])

# --- Create new Onset ---
onset = STALTAOnset(position="classic", sampling_rate=50)
onset.phases = ["P", "S"]
onset.bandpass_filters = {"P": [2, 16, 2], "S": [2, 14, 2]}
onset.sta_lta_windows = {"P": [0.2, 1.0], "S": [0.2, 1.0]}

# --- Create new QuakeScan ---
scan = QuakeScan(
    archive,
    lut,
    onset=onset,
    run_path=run_path,
    run_name=run_name,
    log=True,
    loglevel="info",
)

# --- Set detect parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.timestep = 120.0
# NOTE: increase the thread-count as your system allows. The core migration
# routines are compiled against OpenMP, so the compute time (particularly for
# detect), will decrease roughly linearly with the number of threads used.
scan.threads = 4

# --- Run detect ---
scan.detect(starttime, endtime)

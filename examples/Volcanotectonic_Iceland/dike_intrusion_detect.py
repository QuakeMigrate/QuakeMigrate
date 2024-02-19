# -*- coding: utf-8 -*-
"""
This script runs the detect stage for the Iceland dike intrusion example.

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
station_file = "./inputs/iceland_stations.txt"
data_in = "./inputs/mSEED"
lut_file = "./outputs/lut/dike_intrusion.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run detect ---
starttime = "2014-08-24T00:01:00.0"
endtime = "2014-08-24T00:11:00.0"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Create new Archive and set path structure ---
archive = Archive(
    archive_path=data_in, stations=stations, archive_format="YEAR/JD/STATION"
)

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)
lut.decimate([2, 2, 2], inplace=True)

# --- Create new Onset ---
onset = STALTAOnset(position="classic", sampling_rate=50)
onset.phases = ["P", "S"]
onset.bandpass_filters = {"P": [2, 16, 2], "S": [2, 16, 2]}
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
scan.timestep = 300.0
scan.threads = 4  # NOTE: increase as your system allows to increase speed!

# --- Run detect ---
scan.detect(starttime, endtime)

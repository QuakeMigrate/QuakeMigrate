# -*- coding: utf-8 -*-
"""
This script will run the detect stage of QuakeMigrate.

For more details, please see the manual and read the docs.

"""

from QMigrate.io import Archive, read_lut, read_stations
from QMigrate.signal import QuakeScan
from QMigrate.signal.onset import STALTAOnset

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
archive = Archive(archive_path=archive_path, stations=stations,
                  archive_format="YEAR/JD/STATION")
# For custom structures...
# archive.format = "custom/archive_{year}_{jday}/{month:02d}-{day:02d}.{station}_structure"

# --- Resample data with mismatched sampling rates ---
# archive.resample = True
# archive.upfactor = 2

# --- Load the LUT ---
lut = read_lut(lut_file=lut_file)

# --- Decimate the lookup table ---
lut = lut.decimate([5, 5, 4])

# --- Create new Onset ---
onset = STALTAOnset(position="classic")
onset.p_bp_filter = [2, 9.9, 2]
onset.s_bp_filter = [2, 9.9, 2]
onset.p_onset_win = [0.2, 1.5]
onset.s_onset_win = [0.2, 1.5]

# --- Create new QuakeScan ---
scan = QuakeScan(archive, lut, onset=onset, run_path=run_path,
                 run_name=run_name, log=True, loglevel="info")

# --- Set detect parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.sampling_rate = 20
scan.timestep = 120.
scan.threads = 12

# --- Run detect ---
scan.detect(starttime, endtime)

# -*- coding: utf-8 -*-
"""
Detect stage for the Rutford icequake example.

"""

from QMigrate.io import Archive, read_stations
from QMigrate.lut import LUT
from QMigrate.signal import QuakeScan
from QMigrate.signal.onset import STALTAOnset

# --- i/o paths ---
station_file = "./inputs/rutford_stations.txt"
data_in = "./inputs/mSEED"
lut_out = "./outputs/lut/icequake.LUT"
run_path = "./outputs/runs"
run_name = "icequake_example"

# --- Set time period over which to run detect ---
starttime = "2009-01-21T04:00:05.0"
endtime = "2009-01-21T04:00:15.0"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Create new Archive and set path structure ---
archive = Archive(stations=stations, archive_path=data_in)
archive.path_structure(archive_format="YEAR/JD/*_STATION_*")

# --- Load the LUT ---
lut = LUT(lut_file=lut_out)

# --- Create new Onset ---
onset = STALTAOnset(position="classic")
onset.p_bp_filter = [20, 200, 4]
onset.s_bp_filter = [10, 125, 4]
onset.p_onset_win = [0.01, 0.25]
onset.s_onset_win = [0.05, 0.5]

# --- Create new QuakeScan ---
scan = QuakeScan(archive, lut, onset=onset, run_path=run_path, run_name=run_name)

# --- Set detect parameters ---
scan.sampling_rate = 1000
scan.timestep = 0.75
scan.threads = 12

# --- Run detect ---
scan.detect(starttime, endtime)

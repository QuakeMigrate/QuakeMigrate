# -*- coding: utf-8 -*-
"""
Detect stage for the Iceland icequake example.

"""

import QMigrate.io as qio
from QMigrate.lut import LUT
from QMigrate.signal import QuakeScan
from QMigrate.signal.onset import ClassicSTALTAOnset

# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
data_in = "./inputs/mSEED"
lut_out = "./outputs/LUT/icequake.LUT"
out_path = "./outputs/runs"
run_name = "icequake_example"

# --- Set time period over which to run detect ---
start_time = "2014-06-29T18:41:55.0"
end_time = "2014-06-29T18:42:20.0"

# --- Read in station file ---
stations = qio.stations(station_file)

# --- Create new Archive and set path structure ---
data = qio.Archive(stations=stations, archive_path=data_in)
data.path_structure(archive_format="YEAR/JD/*_STATION_*")

# --- Load the LUT ---
lut = LUT(lut_file=lut_out)

# --- Create new Onset ---
onset = ClassicSTALTAOnset()
onset.p_bp_filter = [10, 125, 4]
onset.s_bp_filter = [10, 125, 4]
onset.p_onset_win = [0.01, 0.25]
onset.s_onset_win = [0.05, 0.5]

# --- Create new QuakeScan ---
scan = QuakeScan(data, lut, onset=onset, output_path=out_path, run_name=run_name)

# --- Set detect parameters ---
scan.sampling_rate = 500
scan.time_step = 0.75
scan.n_cores = 12

# --- Run detect ---
scan.detect(start_time, end_time)

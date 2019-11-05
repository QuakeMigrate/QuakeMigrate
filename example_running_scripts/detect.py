# -*- coding: utf-8 -*-
"""
This script will run the detect stage of QuakeMigrate.

For more details, please see the manual and read the docs.

"""

import QMigrate.io.data as qdata
import QMigrate.lut.lut as qlut
import QMigrate.signal.onset.staltaonset as qonset
import QMigrate.signal.scan as qscan

# --- i/o paths ---
archive_path = "/path/to/archived/data"
lut_file = "/path/to/lut"
out_path = "/path/to/output"
run_name = "name_of_run"
station_file = "/path/to/station_file"

# --- Set time period over which to run detect ---
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.0"

# --- Create new Archive and set path structure ---
data = qdata.Archive(station_file=station_file, archive_path=archive_path)
data.path_structure(archive_format="YEAR/JD/STATION")
# For custom structures...
# data.format = "custom/archive_{year}_{month}_structure"

# --- Resample data with mismatched sampling rates ---
# data.resample = True
# data.upfactor = 2

# --- Load the LUT ---
lut = qlut.LUT(lut_file=lut_file)

# --- Decimate the lookup table ---
lut = lut.decimate([5, 5, 4])

# --- Create new Onset ---
onset = qonset.ClassicSTALTAOnset()
onset.p_bp_filter = [2, 9.9, 2]
onset.s_bp_filter = [2, 9.9, 2]
onset.p_onset_win = [0.2, 1.5]
onset.s_onset_win = [0.2, 1.5]

# --- Create new QuakeScan ---
scan = qscan.QuakeScan(data, lut, onset=onset, output_path=out_path,
                       run_name=run_name, log=False)

# --- Set detect parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.sampling_rate = 20
scan.time_step = 120.
scan.n_cores = 12

# --- Run detect ---
scan.detect(start_time, end_time)

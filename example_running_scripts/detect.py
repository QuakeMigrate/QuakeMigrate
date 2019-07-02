# -*- coding: utf-8 -*-
"""
This script will run the detect stage of QuakeMigrate.

"""

# Import required modules
import QMigrate.signal.scan as qscan
import QMigrate.io.data as qdata

# Set i/o paths
lut_path = "/path/to/lut"
data_path = "/path/to/data"
station_file = "/path/to/station_file"
out_path = "/path/to/output"
run_name = "name_of_run"

# Time period over which to run detect
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.0"

# Create a new instance of Archive and set path structure
data = qdata.Archive(station_file=station_file, archive_path=data_path)
data.path_structure(archive_format="YEAR/JD/STATION")

# Resample data with mismatched sampling rates
# data.resample = True
# data.upfactor = 2

# Create a new instance of SeisScan object
scan = qscan.QuakeScan(data, lut_path, output_path=out_path, run_name=run_name)

# Set detect parameters - for a complete list and guidance on how to choose
# a suitable set of parameters, please consult the documentation
scan.sampling_rate = 20
scan.p_bp_filter = [2, 9.9, 2]
scan.s_bp_filter = [2, 9.9, 2]
scan.p_onset_win = [0.2, 1.5]
scan.s_onset_win = [0.2, 1.5]
scan.time_step = 120.
scan.decimate = [5, 5, 4]
scan.n_cores = 12

# Run detect
scan.detect(start_time, end_time)

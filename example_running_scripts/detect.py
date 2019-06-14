# -*- coding: utf-8 -*-
"""
This script will run the detect stage of QuakeMigrate.

Author: Conor Bacon
"""

# Import required modules
import QMigrate.signal.scan as qscan
import QMigrate.io.mseed as qmseed

# Set i/o paths
lut_path = "/path/to/lut"
data_path = "/path/to/data"
out_path = "/path/to/output"
run_name = "name_of_run"

# Create a new instance of MSEED and set path structure
data = qmseed.MSEED(lut_path, HOST_PATH=data_path)
data.path_structure(path_type="YEAR/JD/STATION")

# Create a new instance of SeisScan object
scn = qscan.SeisScan(data, lut_path, output_path=out_path, output_name=run_name)

# Set detect parameters - for a complete list and guidance on how to choose
# a suitable set of parameters, please consult the documentation
scn.sampling_rate = 20
scn.p_bp_filter = [2, 9.9, 2]
scn.s_bp_filter = [2, 9.9, 2]
scn.p_onset_win = [0.2, 1.5]
scn.s_onset_win = [0.2, 1.5]
scn.time_step = 120.
scn.decimate = [5, 5, 4]
scn.n_cores = 12

# Time period over which to run detect
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.0"

# Run detect
scn.detect(start_time, end_time)


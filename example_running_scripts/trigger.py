# -*- coding: utf-8 -*-

"""
This script will run the trigger and locate stages of QuakeMigrate.

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

# Create a new instance of MSEED and set path structure (default)
data = qmseed.MSEED(lut_path, HOST_PATH=data_path)
data.path_structure(path_type="YEAR/JD/STATION")

# Create a new instance of SeisScan object
scn = qscan.SeisScan(data, lut_path, output_path=out_path, output_name=run_name)

# Set trigger/locate parameters
scn.sampling_rate = 20
scn.p_bp_filter = [2, 9.9, 2]
scn.s_bp_filter = [2, 9.9, 2]
scn.p_onset_win = [0.2, 1.]
scn.s_onset_win = [0.2, 1.]
scn.time_step = 120.
scn.decimate = [1, 1, 1]
scn.n_cores = 12
scn.normalise_coalescence = True
scn.detection_threshold = 1.75
scn.marginal_window = 1
scn.minimum_repeat = 30.0

# Turn on plotting features
scn.plot_coal_trace = True
scn.plot_coal_picture = True
scn.plot_coal_video = False

# Time period over which to run trigger and locate
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.0"

# Run trigger
scn.trigger(start_time, end_time)

# Run locate
scn.locate(start_time, end_time, cut_mseed=True)

# -*- coding: utf-8 -*-
"""
This script will run the trigger stage of QuakeMigrate.

For more details, please see the manual and read the docs.

"""

import QMigrate.io as qio
from QMigrate.signal import Trigger

# --- i/o paths ---
out_path = "/path/to/output"
run_name = "name_of_run"
station_file = "/path/to/station_file"

# --- Set time period over which to run trigger ---
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.00"

# --- Read in station file ---
stations = qio.stations(station_file)

# --- Create new Trigger ---
trig = Trigger(out_path, run_name, stations)

# --- Set trigger parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
trig.marginal_window = 1.
trig.minimum_repeat = 30.
trig.normalise_coalescence = True

# --- Static threshold ---
trig.detection_threshold = 1.75

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.detection_threshold = "dynamic"
# trig.mad_window_length = 7200.
# trig.mad_multiplier = 8.

# --- Run trigger ---
trig.trigger(start_time, end_time, savefig=False)

# -*- coding: utf-8 -*-
"""
Trigger stage for the Iceland icequake example.

"""

import QMigrate.io as qio
from QMigrate.signal import Trigger

# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
out_path = "./outputs/runs"
run_name = "icequake_example"

# --- Set time period over which to run trigger ---
start_time = "2014-06-29T18:41:55.0"
end_time = "2014-06-29T18:42:20.0"

# --- Read in station file ---
stations = qio.stations(station_file)

# --- Create new Trigger ---
trig = Trigger(out_path, run_name, stations)

# --- Set trigger parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
trig.marginal_window = 2.75
trig.minimum_repeat = 6.
trig.normalise_coalescence = True

# --- Static threshold ---
trig.detection_threshold = 1.8

# --- Dynamic (Median Absolute Deviation) threshold ---
# trig.detection_threshold = "dynamic"
# trig.mad_window_length = 7200.
# trig.mad_multiplier = 8.

# --- Run trigger ---
trig.trigger(start_time, end_time, savefig=False)

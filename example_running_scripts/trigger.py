# -*- coding: utf-8 -*-
"""
This script will run the trigger stage of QuakeMigrate.

"""

# Import required modules
import QMigrate.signal.trigger as qtrigger
import QMigrate.io.quakeio as qio

# Set i/o paths
station_file = "/path/to/station_file"
out_path = "/path/to/output"
run_name = "name_of_run"

# Time period over which to run trigger
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.00"

# Read in station files
stations = qio.stations(station_file)

# Create a new instance of Trigger
trig = qtrigger.Trigger(out_path, run_name, stations)

# Set trigger parameters - for a complete list and guidance on how to choose
# a suitable set of parameters, please consult the documentation
trig.normalise_coalescence = True
trig.marginal_window = 1.
trig.minimum_repeat = 30.
trig.detection_threshold = 1.75

# Run trigger
trig.trigger(start_time, end_time, savefig=False)

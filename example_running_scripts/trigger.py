# -*- coding: utf-8 -*-
"""
This script will run the trigger stage of QuakeMigrate.

"""

import QMigrate.signal.trigger as qtrigger

# Set i/o paths
out_path = "./outputs/runs"
run_name = "name_of_run"

# Set trigger/locate parameters
sampling_rate = 50
normalise_coalescence = True
detection_threshold = 1.85
marginal_window = 1
minimum_repeat = 30

# Time period over which to run trigger
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.00"

# Read in station files
stations = qtrigger.stations("./inputs/stations/stations.in",
                             units="lat_lon_elev")

# Run trigger
qtrigger.trigger(start_time, end_time, out_path, run_name, marginal_window,
                 detection_threshold, normalise_coalescence, minimum_repeat,
                 sampling_rate, stations, savefig=True)

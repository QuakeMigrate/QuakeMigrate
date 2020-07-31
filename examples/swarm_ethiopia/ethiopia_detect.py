# -*- coding: utf-8 -*-
"""
Detect stage for the ethiopia swarm example.

"""

from quakemigrate.io import Archive, read_lut, read_stations
from quakemigrate.signal import QuakeScan
from quakemigrate.signal.onset import STALTAOnset

# --- i/o paths ---
station_file = "./inputs/ethiopia_stations_TM.csv"
data_in = "./inputs/mSEED/"
lut_out = "./outputs/lut/ethiopia.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run detect ---
starttime = "2016-097T18:31:00"
endtime = "2016-097T18:44:00"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Create new Archive and set path structure ---
archive = Archive(archive_path=data_in, stations=stations,
                  format='{year}/{jday:03d}/{network}.{station}*',
                  catch_network=True)

# --- Load the LUT ---
lut = read_lut(lut_file=lut_out)
lut = lut.decimate([5, 5, 5])

# --- Create new Onset ---
onset = STALTAOnset(position="classic")
onset.p_bp_filter = [1., 10., 2]
onset.s_bp_filter = [1., 10., 2]
onset.p_onset_win = [0.7, 5.]
onset.s_onset_win = [0.7, 5.]

# --- Create new QuakeScan ---
scan = QuakeScan(archive, lut, onset=onset, run_path=run_path,
                 run_name=run_name, log=True, loglevel="debug")

# --- Set detect parameters ---
scan.sampling_rate = 50
scan.timestep = 60.
scan.threads = 20

# --- Run detect ---
scan.detect(starttime, endtime)

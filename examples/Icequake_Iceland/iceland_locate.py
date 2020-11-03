# -*- coding: utf-8 -*-
"""
Locate stage for the Iceland icequake example.

"""

from quakemigrate import QuakeScan
from quakemigrate.io import Archive, read_lut, read_stations
from quakemigrate.signal.onsets import STALTAOnset
from quakemigrate.signal.pickers import GaussianPicker

# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
data_in = "./inputs/mSEED"
lut_out = "./outputs/lut/example.LUT"
run_path = "./outputs/runs"
run_name = "example_run"

# --- Set time period over which to run locate ---
starttime = "2014-06-29T18:41:55.0"
endtime = "2014-06-29T18:42:20.0"

# --- Read in station file ---
stations = read_stations(station_file)

# --- Create new Archive and set path structure ---
archive = Archive(archive_path=data_in, stations=stations,
                  archive_format="YEAR/JD/*_STATION_*")

# --- Load the LUT ---
lut = read_lut(lut_file=lut_out)

# --- Create new Onset ---
onset = STALTAOnset(position="centred", sampling_rate=500)
onset.bandpass_filters = {
    "P": [10, 125, 4],
    "S": [10, 125, 4]}
onset.sta_lta_windows = {
    "P": [0.01, 0.25],
    "S": [0.05, 0.5]}

# --- Create new PhasePicker ---
picker = GaussianPicker(onset=onset, plot_picks=True)
picker.sampling_rate = 500

# --- Create new QuakeScan ---
scan = QuakeScan(archive, lut, onset=onset, picker=picker,
                 run_path=run_path, run_name=run_name, log=True,
                 loglevel="info")

# --- Set locate parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.marginal_window = 1.
scan.threads = 12
scan.scan_rate = 500

# --- Toggle plotting options ---
scan.plot_event_video = False
scan.plot_event_summary = True

# --- Toggle writing of waveforms ---
scan.write_cut_waveforms = True

# --- Run locate ---
scan.locate(starttime=starttime, endtime=endtime)

# -*- coding: utf-8 -*-
"""
Locate stage for the Iceland icequake example.

"""


import QMigrate.io as qio
from QMigrate.lut import LUT
from QMigrate.signal import QuakeScan
from QMigrate.signal.onset import CentredSTALTAOnset
from QMigrate.signal.pick import GaussianPicker

# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
data_in = "./inputs/mSEED"
lut_out = "./outputs/LUT/icequake.LUT"
out_path = "./outputs/runs"
run_name = "icequake_example"

# --- Set time period over which to run locate ---
start_time = "2014-06-29T18:41:55.0"
end_time = "2014-06-29T18:42:20.0"

# --- Read in station file ---
stations = qio.stations(station_file)

# --- Create new Archive and set path structure ---
data = qio.Archive(stations=stations, archive_path=data_in)
data.path_structure(archive_format="YEAR/JD/*_STATION_*")

# --- Load the LUT ---
lut = LUT(lut_file=lut_out)

# --- Create new Onset ---
onset = CentredSTALTAOnset()
onset.p_bp_filter = [10, 125, 4]
onset.s_bp_filter = [10, 125, 4]
onset.p_onset_win = [0.01, 0.25]
onset.s_onset_win = [0.05, 0.5]

# --- Create new PhasePicker ---
picker = GaussianPicker(onset=onset)
picker.marginal_window = 2.75
picker.plot_phase_picks = True

# --- Create new QuakeScan ---
scan = QuakeScan(data, lut, onset=onset, picker=picker,
                 output_path=out_path, run_name=run_name, log=True)

# --- Set locate parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.marginal_window = 2.75
scan.n_cores = 12
scan.sampling_rate = 500

# --- Toggle plotting options ---
scan.plot_event_video = False
scan.plot_event_summary = True
scan.plot_station_traces = True

# --- Toggle writing of waveforms ---
scan.write_cut_waveforms = True

# --- Run locate ---
scan.locate(start_time=start_time, end_time=end_time)

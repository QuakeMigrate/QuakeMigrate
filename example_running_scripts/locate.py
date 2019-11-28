# -*- coding: utf-8 -*-
"""
This script will run the locate stage of QuakeMigrate.

For more details, please see the manual and read the docs.

"""

import QMigrate.io.data as qdata
import QMigrate.lut.lut as qlut
import QMigrate.signal.onset.staltaonset as qonset
import QMigrate.signal.pick.gaussianpicker as qpick
import QMigrate.signal.scan as qscan

# --- i/o paths ---
lut_path = "/path/to/lut"
data_path = "/path/to/data"
station_file = "/path/to/station_file"
out_path = "/path/to/output"
run_name = "name_of_run"

# --- Set time period over which to run locate ---
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.0"

# --- Create new Archive and set path structure ---
data = qdata.Archive(station_file=station_file, archive_path=data_path)
data.path_structure(archive_format="YEAR/JD/STATION")
# For custom structures...
# data.format = "custom/archive_{year}_{month}_structure"

# --- Resample data with mismatched sampling rates ---
# data.resample = True
# data.upfactor = 2

# --- Load the LUT ---
lut = qlut.LUT(lut_file=lut_path)

# --- Decimate the lookup table ---
# lut = lut.decimate([1, 1, 1])

# --- Create new Onset ---
onset = qonset.CentredSTALTAOnset()
onset.p_bp_filter = [2, 9.9, 2]
onset.s_bp_filter = [2, 9.9, 2]
onset.p_onset_win = [0.2, 1.5]
onset.s_onset_win = [0.2, 1.5]

# --- Create new PhasePicker ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
gausspicker_onset = qonset.CentredSTALTAOnset()
gausspicker_onset.p_bp_filter = [2, 9.9, 2]
gausspicker_onset.s_bp_filter = [2, 9.9, 2]
gausspicker_onset.p_onset_win = [0.2, 1.5]
gausspicker_onset.s_onset_win = [0.2, 1.5]

picker = qpick.GaussianPicker(onset=gausspicker_onset)
picker.marginal_window = 1
picker.plot_phase_picks = True

# --- Create new QuakeScan ---
scan = qscan.QuakeScan(data, lut, onset=onset, picker=picker,
                       output_path=out_path, run_name=run_name, log=True)


# --- Set locate parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.sampling_rate = 20
scan.n_cores = 12
scan.marginal_window = 1

# --- Toggle plotting options ---
scan.plot_event_summary = True
scan.plot_coal_video = False

# --- Toggle writing of waveforms ---
scan.write_cut_waveforms = True

# --- Run locate ---
scan.locate(start_time=start_time, end_time=end_time)
# scan.locate(fname="filename_of_triggered_events")

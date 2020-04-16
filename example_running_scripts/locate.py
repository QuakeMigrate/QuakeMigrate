# -*- coding: utf-8 -*-
"""
This script will run the locate stage of QuakeMigrate.

For more details, please see the manual and read the docs.

"""

import QMigrate.io as qio
from QMigrate.lut import LUT
from QMigrate.signal import QuakeScan
from QMigrate.signal.onset import CentredSTALTAOnset
from QMigrate.signal.pick import GaussianPicker

# --- i/o paths ---
archive_path = "/path/to/archived/data"
lut_file = "/path/to/lut"
station_file = "/path/to/station_file"
out_path = "/path/to/output"
run_name = "name_of_run"

# --- Set time period over which to run locate ---
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.0"

# --- Read in station file ---
stations = qio.stations(station_file)

# --- Create new Archive and set path structure ---
data = qio.Archive(stations=stations, archive_path=archive_path)
data.path_structure(archive_format="YEAR/JD/STATION")
# For custom structures...
# data.format = "custom/archive_{year}_{month}_structure"

# --- Resample data with mismatched sampling rates ---
# data.resample = True
# data.upfactor = 2

# --- Specify parameters for amplitude measurement ---
amp_params = {'pre_filt' : (0.02, 0.03, 20, 25),         # pre-filter used if doing full instrument response removal
              'water_level' : 600,                       # water-level used if doing full instrument response removal
              'response_fname' : 'Z7_dataless.xml',      # path to file containing instrument response information; dataless.SEED, stationXML, concatenated RESP files, etc.
              'noise_win' : 10.,                         # length of the noise window; to make a measurement of the background noise before earthquake phase arrivals
              'signal_window' : 5.,                      # length of the S-wave signal window
              'prominence_multiplier': 0.,               # parameter passed to find_peaks (0. is recommended)
              'highpass_filter' : True,                  # whether to apply a high-pass filter before measuring amplitudes
              'highpass_freq' : 2.,                      # high-pass filter frequency
              'bandpass_filter' : False,                 # whether to apply a band-pass filter before measuring amplitudes
              'bandpass_lowcut' : 2.,                    # band-pass filter low-cut frequency 
              'bandpass_highcut' : 20.,                  # band-pass filter high-cut frequency
              'filter_corners': 4,                       # number of corners for the chosen filter
              'remove_FIR_response' : False}             # whether to remove all response stages, inc. FIR (st.remove_response()), not just poles-and-zero response stage; significantly slower

# --- Specify parameters for magnitude calculation ---
mag_params = {'station_corrections' : {},                # dictionary of trace_id - correction pairs
              'amplitude_feature' : 'S_amp',             # which amplitude feature to calculate magnitudes from (S_amp or P_amp)
              'use_hyp_distance' : True,                 # use hypocentral rather than epicentral distance
              'A0' : 'Hutton-Boore',                     # which A0 attenuation correction to apply, a function can be directly passed here
              'trace_filter' : '.[BH]H[NE]$',            # which traces to calculate magnitude from
              'dist_filter' : False,                     # filter magnitudes based on event-station distance
              'use_only_picked' : False,                 # use only amplitude observations from traces which have been picked by the autopicker
              'noise_filter' : 2.,                       # a factor to multiply the noise by to remove traces with high noise levels
              'weighted' : False}                        # do a weighted mean of the magnitudes

# --- Load the LUT ---
lut = LUT(lut_file=lut_file)

# --- Decimate the lookup table ---
# lut = lut.decimate([1, 1, 1])

# --- Create new Onset ---
onset = CentredSTALTAOnset()
onset.p_bp_filter = [2, 9.9, 2]
onset.s_bp_filter = [2, 9.9, 2]
onset.p_onset_win = [0.2, 1.5]
onset.s_onset_win = [0.2, 1.5]

# --- Create new PhasePicker ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
gausspicker_onset = CentredSTALTAOnset()
gausspicker_onset.p_bp_filter = [2, 9.9, 2]
gausspicker_onset.s_bp_filter = [2, 9.9, 2]
gausspicker_onset.p_onset_win = [0.2, 1.5]
gausspicker_onset.s_onset_win = [0.2, 1.5]

picker = GaussianPicker(onset=gausspicker_onset)
picker.marginal_window = 1
picker.plot_phase_picks = True

# --- Create new QuakeScan ---
scan = QuakeScan(data, lut, onset=onset, picker=picker,
                 output_path=out_path, run_name=run_name, log=True,
                 get_amplitudes=False, calc_mag=False,
                 quick_amplitudes=False)

# --- Set locate parameters ---
# For a complete list of parameters and guidance on how to choose them, please
# see the manual and read the docs.
scan.marginal_window = 1
scan.n_cores = 12
scan.sampling_rate = 20

# --- Set amplitude / magnitude parameters ---
scan.amplitude_params = amp_params
scan.magnitude_params = mag_params

# --- Toggle plotting options ---
scan.plot_event_video = False
scan.plot_event_summary = True
scan.plot_station_traces = True

# --- Toggle writing of cut waveforms ---
scan.write_cut_waveforms = True
scan.pre_cut = 20.
scan.post_cut = 60.

# --- Run locate ---
scan.locate(start_time=start_time, end_time=end_time)
# scan.locate(fname="filename_of_triggered_events")

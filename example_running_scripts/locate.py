# -*- coding: utf-8 -*-
"""
This script will run the locate stage of QuakeMigrate.

"""

# Import required modules
import QMigrate.core.model as qmod
import QMigrate.io.data as qdata
import QMigrate.signal.onset.staltaonset as qonset
import QMigrate.signal.pick.gaussianpicker as qpick
import QMigrate.signal.scan as qscan

# Set i/o paths
lut_path = "/path/to/lut"
data_path = "/path/to/data"
station_file = "/path/to/station_file"
out_path = "/path/to/output"
run_name = "name_of_run"

# Time period over which to run locate
start_time = "2018-001T00:00:00.0"
end_time = "2018-002T00:00:00.0"

# Create a new instance of Archive and set path structure
data = qdata.Archive(station_file=station_file, archive_path=data_path)
data.path_structure(archive_format="YEAR/JD/STATION")

# extract the amplitude information and magnitude
amp_param = {'pre_filt' : None, # pre-filter used in the instrument correction
            'water_level' : 30, # water-level used in the instrument correction
            'dataless_filt' : 'path_to_dataless_SEED', # path to dataless SEED volume
            'noise_win' : 10.} # length of the noise and S-wave window
mag_param = {'station_corrections' : {}, # dictionary of trace_id - correction paire
            'amplitude_feature' : 'S_amp', # which amplitude feature to make the calculation from
            'use_hyp_distance' : True, # use hypocentral rather than epicentral distance
            'A0' : 'keir2006', # which A0 attenuation correction to apply, a function can be directly passed here
            'trace_filter' : '.[BH]H[NE]$', # which traces to calculate magnitude from 
            'dist_filter' : False, # filter based on distance
            'use_only_picked' : False, # use only those traces which have been picked by the autopicker
            'noise_filter' : 4., # a factor to multiply the noise by to remove traces with high noise levels
            'weighted' : True} # do a weighted mean

# Resample data with mismatched sampling rates
# data.resample = True
# data.upfactor = 2

# Load the LUT
lut = qmod.LUT()
lut.load(lut_path)

# Decimate the lookup table in each dimension
lut = lut.decimate([1, 1, 1])

# Create a new instance of Onset object
onset = qonset.CentredSTALTAOnset()
onset.p_bp_filter = [2, 9.9, 2]
onset.s_bp_filter = [2, 9.9, 2]
onset.p_onset_win = [0.2, 1.5]
onset.s_onset_win = [0.2, 1.5]

# Create a new instance of PhasePicker object
# The Gaussian picker uses the STA/LTA function, but different parameters may
# be used.
gausspicker_onset = qonset.CentredSTALTAOnset()
gausspicker_onset.p_bp_filter = [2, 9.9, 2]
gausspicker_onset.s_bp_filter = [2, 9.9, 2]
gausspicker_onset.p_onset_win = [0.2, 1.5]
gausspicker_onset.s_onset_win = [0.2, 1.5]

picker = qpick.GaussianPicker(onset=gausspicker_onset)
picker.marginal_window = 1
picker.plot_phase_picks = True

# Create a new instance of QuakeScan object
scan = qscan.QuakeScan(data, lut, onset=onset, picker=picker,
                       output_path=out_path, run_name=run_name, log=True)

# Set locate parameters - for a complete list and guidance on how to choose
# a suitable set of parameters, please consult the documentation
scan.sampling_rate = 20
scan.n_cores = 12
scan.marginal_window = 1

# Turn on plotting features
scan.plot_event_summary = True
scan.plot_coal_video = False

# Output cut data
scan.write_cut_waveforms = True

# turn on magnitude and amplitude extraction
scan.amplitudes = True
scan.quicks_amps = False
scan.amplitude_parameters = amp_param
scan.mag_calc = True
scan.magnitude_parameters = mag_param

# Run locate
scan.locate(start_time=start_time, end_time=end_time)
# scan.locate(fname="filename_of_triggered_events")

#!/usr/bin/env python
# coding: utf-8

# # QuakeMigrate example - Icequake detection at the Rutford icestream

# ## Overview

# This notebook contains another example showing how to run QuakeMigrate for icequake detection.
# 
# Here, we detail how to:
# 1. Create travel-time lookup tables for the example seismometer network
# 2. Run the detect stage to coalesce energy through time
# 3. Run the trigger stage to determine events above a threshold value
# 4. Run the locate stage to refine the earthquake location

# In[1]:


# Import necessary modules:
from pyproj import Proj

import pandas as pd

import QMigrate.io.data as qdata
import QMigrate.io.quakeio as qio
import QMigrate.lut.lut as qlut
import QMigrate.signal.onset.staltaonset as qonset
import QMigrate.signal.pick.gaussianpicker as qpick
import QMigrate.signal.scan as qscan
import QMigrate.signal.trigger as qtrigger


# In[2]:


# Set i/o paths:
station_file = "./inputs/rutford_stations.txt"
data_in   = "./inputs/mSEED"
lut_out   = "./outputs/lut/icequake.LUT"
out_path  = "./outputs/runs"
run_name  = "icequake_example"


# ## 1. Create travel-time lookup tables (LUTs)

# In[16]:


import QMigrate.lut.create_lut as clut

# Read in station information
stations = qio.stations(station_file)

# Define projections
cproj = Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
gproj = Proj("+proj=lcc +lon_0=-83.932 +lat_0=-78.144 +lat_1=-78.1 +lat_2=-77.9 +datum=WGS84 +units=m +no_defs")

# Set the parameters for the travel-times lookup table (LUT)
# Cell count (x,y,z); cell size (x,y,z in metres)
lut = qlut.LUT(ll_corner=[-84.14853353566141, -78.18825429331356, -350.],
               ur_corner=[-83.71921885073093, -78.10003166259442, 3550],
               cell_size=[100., 100., 100.], grid_proj=gproj, coord_proj=cproj)

# Compute for a homogeneous velocity model
vp = 3841
vs = 1970
clut.compute(lut, stations, method="homogeneous", vp=vp, vs=vs)

# Save the LUT
lut.save(lut_out)


# In[18]:


lut.cell_count


# ## 2. Coalesce the seismic energy through time

# In[22]:


# Create a new instance of the MSEED class and set path structure
data = qdata.Archive(station_file=station_file, archive_path=data_in)
data.path_structure(archive_format="YEAR/JD/*_STATION_*")

# Create a new instance of Onset object
onset = qonset.ClassicSTALTAOnset()
onset.p_bp_filter = [20, 200, 4]
onset.s_bp_filter = [10, 125, 4]
onset.p_onset_win = [0.01, 0.25]
onset.s_onset_win = [0.05, 0.5]

# Create a new instance of the SeisScan class
scan = qscan.QuakeScan(data, lut, onset=onset, output_path=out_path, run_name=run_name)

# Set detect parameters
scan.sampling_rate = 1000
scan.time_step = 0.75
scan.n_cores = 12

# Defining the start and end times 
starttime = "2009-01-21T04:00:05.0"
endtime   = "2009-01-21T04:00:15.0"


# In[23]:


# Run the detect stage to find the coalescence of energy through time:
scan.detect(starttime, endtime)


# ## 3. Run the trigger stage, to detect and output individual icequakes
# 
# nb: We can use the same SeisScan object here because we are not using a different decimation. If running trigger and locate on grids with different levels of decimation, a new SeisScan object must be initialised.

# In[21]:


trig = qtrigger.Trigger(out_path, run_name, stations)

trig.normalise_coalescence = True
trig.marginal_window = 0.1
trig.minimum_repeat = 0.5
trig.detection_threshold = 2.75

# Run trigger
trig.trigger(starttime, endtime, savefig=True)


# ## 4. Run the locate stage, to relocate triggered events on a less decimated grid

# In[27]:


# Create a new instance of PhasePicker object
picker = qpick.GaussianPicker(onset=onset)
picker.marginal_window = 0.1
picker.plot_phase_picks = True

# Create a new instance of QuakeScan object
scan = qscan.QuakeScan(data, lut, onset=onset, picker=picker,
                       output_path=out_path, run_name=run_name, log=True)

# Set locate parameters:
scan.sampling_rate = 1000
scan.marginal_window = 0.1
scan.n_cores = 12

# Turn on plotting features
scan.plot_event_summary = True
scan.plot_event_video = False
scan.write_cut_waveforms = False


# In[28]:


# Run the locate stage to determine the location of any triggered events
scan.locate(start_time=starttime, end_time=endtime)


# ## 5. Some of the key outputs

# In[30]:


# Show the .event file, containing event origin time and location:
icequake_event_fname = "./outputs/runs/icequake_example/locate/events/20090121040007152.event"
event_df = pd.read_csv(icequake_event_fname)

event_df


# In[31]:


# Show the .picks file, containing station time picks:
icequake_pick_fname = "outputs/runs/icequake_example/locate/picks/20090121040007152.picks"
pick_df = pd.read_csv(icequake_pick_fname)

pick_df


# In[33]:


# Show the coalescence pdf file, containing event origin time and location:
icequake_coal_image_fname = "outputs/runs/icequake_example/locate/summaries/icequake_example_20090121040007152_EventSummary.pdf"
from IPython.display import IFrame # For plotting pdf
IFrame(icequake_coal_image_fname, width=800, height=400) # Plot pdf


# In[ ]:





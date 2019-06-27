#!/usr/bin/env python
# coding: utf-8

# # QuakeMigrate - Example - Icequake detection

# ## Overview:

# This notebook shows how to run QuakeMigrate for icequake detection, using a 2 minute window of continuous seismic data from Hudson et al (2019). Please refer to this paper for details and justification of the settings used.
# 
# Here, we detail how to:
# 1. Create a travel-times lookup table for the example seismometer network
# 2. Run the detect stage to coalesce energy through time
# 3. Run the trigger stage to determine events above a threshold value
# 4. Run the locate stage to refine the earthquake location
# 
# We also provide an outline of some of the key outputs

# In[33]:


# Import necessary modules:
import QMigrate.core.model as qmod
import QMigrate.signal.scan as qscan
import QMigrate.io.data as qdata
import QMigrate.io.quakeio as qio
import QMigrate.signal.trigger as qtrigger


# In[34]:


# Set i/o paths:
station_file = "./inputs/stations.txt"
data_in   = "./inputs/mSEED"
lut_out   = "./outputs/lut/icequake.LUT"
out_path  = "./outputs/runs"
run_name  = "icequake_example"


# ## 1. Create a travel-times lookup table (LUT)

# In[35]:


# Read in station information
stations = qio.stations(station_file)

# Set the parameters for the travel-times lookup table (LUT)
# Cell count (x,y,z); cell size (x,y,z in metres)
lut = qmod.LUT(stations, cell_count=[20, 20, 140], cell_size=[100, 100, 20])
lut.lonlat_centre(-17.224, 64.328)

# Set the LUT projection (here we use the Lambert Conformal Conic projection)
lut.lcc_standard_parallels = (64.32, 64.335)
lut.projections(grid_proj_type="LCC")
lut.elevation=1400 # Defining the elevation of the top of the grid in m 

# Compute for a homogeneous velocity model
v_p_homo_model = 3630
v_s_homo_model = 1833
lut.compute_homogeneous_vmodel(v_p_homo_model, v_s_homo_model)

# Save the LUT
lut.save(lut_out)


# ## 2. Coalesce the seismic energy through time

# In[37]:


# Create a new instance of the MSEED class and set path structure
data = qdata.Archive(station_file=station_file, archive_path=data_in)
data.path_structure(archive_format="YEAR/JD/*_STATION_*")

# Create a new instance of the SeisScan class
scan = qscan.QuakeScan(data, lut_out, output_path=out_path, run_name=run_name)


# In[38]:


# Set detect parameters
scan.sampling_rate = 500           # Sampling rate of data, in Hz
scan.p_bp_filter   = [10, 125, 4]  # The band-pass filter parameters for the P-phase (10 to 125 Hz, with 4th order corners)
scan.s_bp_filter   = [10, 125, 4]  # The band-pass filter parameters for the P-phase (10 to 125 Hz, with 4th order corners)
scan.p_onset_win   = [0.01, 0.25]  # Length of the STA and LTA time windows for the P-phase
scan.s_onset_win   = [0.05, 0.5]   # Length of the STA and LTA time windows for the S-phase
scan.time_step     = 0.75          # The length of the time-step
scan.decimate      = [1, 1, 1]     # Decimation factors in x,y,z (no decimation here)
scan.n_cores       = 12            # Number of cores/processors to use

# Defining the start and end times 
starttime = "2014-06-29T18:41:55.0"
endtime   = "2014-06-29T18:42:20.0"


# In[39]:


# Run the detect stage to find the coalescence of energy through time:
scan.detect(starttime, endtime)


# ## 3. Run the trigger stage, to detect and output individual icequakes
# 
# nb: We can use the same SeisScan object here because we are not using a different decimation. If running trigger and locate on grids with different levels of decimation, a new SeisScan object must be initialised.

# In[41]:


trig = qtrigger.Trigger(out_path, run_name, stations)

trig.normalise_coalescence = True
trig.marginal_window = 2.75
trig.minimum_repeat = 6.
trig.detection_threshold = 1.8

# Run trigger
trig.trigger(starttime, endtime, savefig=True)


# ## 4. Run the locate stage, to relocate triggered events on a less decimated grid

# In[42]:


# Set locate parameters:
scan.marginal_window = 2.75

# Turn on plotting features
scan.plot_coal_video      = False
scan.plot_coal_grid       = False
scan.plot_coal_picture    = True
scan.plot_coal_trace      = False


# In[43]:


# Run the locate stage to determine the location of any triggered events
scan.locate(starttime, endtime)


# ## 4. Some of the key outputs

# In[48]:


# Show the .event file, containing event origin time and location:
icequake_event_fname = "./outputs/runs/icequake_example/events/20140629184210330000.event"
with open(icequake_event_fname) as f:
    lines = f.readlines()
for line in lines:
    print(line)


# In[49]:


# Show the .stn file, containing station time picks:
icequake_stn_fname = "outputs/runs/icequake_example/picks/20140629184210330000.picks"
with open(icequake_stn_fname) as f:
    lines = f.readlines()
for line in lines:
    print(line)


# In[50]:


# Show the coalescence pdf file, containing event origin time and location:
icequake_coal_image_fname = "outputs/runs/icequake_example/summaries/icequake_example_20140629184210330000_EventSummary.pdf"
from IPython.display import IFrame # For plotting pdf
IFrame(icequake_coal_image_fname, width=800, height=400) # Plot pdf


# References:
# 
# Hudson, T.S., Smith, J., Brisbourne, A.M., and White R.S. (2019). Automated detection of basal icequakes and discrimination from surface crevassing. Annals of Glaciology, 79

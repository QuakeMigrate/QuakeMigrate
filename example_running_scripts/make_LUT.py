# -*- coding: utf-8 -*-

"""
This script will create the traveltime lookup tables for QuakeMigrate

Author: Conor Bacon
"""

import QMigrate.core.model as qmod
from pyproj import Proj
import pandas as pd

# Set i/o paths
lut_path = "/path/to/lookup_table"
stat_path = "/path/to/station_file"
vmod_path = "/path/to/vmodel_file"

# Create a new instance of LUT and set parameters
lut = qmod.LUT(cell_count=[300, 300, 60], cell_size=[500, 500, 500])
lut.lonlat_centre(116.75, 6.25)
lut.elevation = 2000.  # Accounts for surface being above sea level
lut.stations(path=stat_path, units="lat_lon_elev")

# --- skfmm LUT generation ---
# ----------------------------
lut.lcc_standard_parallels = (4.0, 7.5)
lut.projections(grid_proj_type="LCC")

# Compute traveltimes for 1-D velocity model
lut.compute_1d_vmodel_skfmm(path=vmod_path)

# ----------------------------

# --- NLLoc sweep LUT generation ---
# ----------------------------------
# Define the cartesian projection as a PyProj object
p0 = Proj("+proj=latlong +ellps=WGS84")
p1 = Proj("+proj=lcc +lat_0=6.7 +lon_0=38.2 +ellps=WGS84 +lat_1=7.2 +lat2=7.8")

# Define the upper, bottom left and the lower, upper right corners and the
# grid spacing of the final QuakeMigrate travel time grid.
# Note that the Z coords and the grid spacing are in metres.
gridspec = [(38.7, 7.4, -5000),   # lon0, lat0, z0
            (38.9, 7.65, 30000),  # lon1, lat1, z1
            (500, 500, 500)]      # dx, dy, dz

# Read in the 1D velocity model
vmodel = pd.read_csv("vmodel.csv", delimiter=",")

lut.compute_1d_vmodel(p0, p1, gridspec, vmodel, block_model=True)
# ----------------------------------

# Output LUT
lut.save(lut_path)

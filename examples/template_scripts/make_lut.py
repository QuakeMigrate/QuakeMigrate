# -*- coding: utf-8 -*-
"""
This script will create travel-time lookup tables for QuakeMigrate.

For more details, please see the manual and read the docs.

"""

from pyproj import Proj

from QMigrate.io import read_stations, read_vmodel
from QMigrate.lut import compute_traveltimes, read_nlloc

# --- i/o paths ---
lut_file = "/path/to/save/lut_file"
station_file = "/path/to/station_file"
vmodel_file = "/path/to/vmodel_file"

# --- Read in the stations and velocity model files ---
stations = read_stations(station_file)
vmod = read_vmodel(vmodel_file)

# --- Define the grid specifications ---
# Values used are for a region in Sabah, Borneo
grid_spec = {"ll_corner": [116.075, 5.573, -1750],
             "ur_corner": [117.426, 6.925, 27750],
             "cell_size": [500., 500., 500.],
             "grid_proj": Proj(proj="lcc", lon_0=116.75, lat_0=6.25, lat_1=4.0,
                               lat_2=7.5, datum="WGS84", ellps="WGS84",
                               units="m", no_defs=True),
             "coord_proj": Proj(proj="longlat", ellps="WGS84", datum="WGS84",
                                no_defs=True)}

# --- Homogeneous LUT generation ---
lut = compute(grid_spec, stations, method="homogeneous", vp=5000., vs=3000.,
              log=True, save_file=lut_file)

# --- skfmm LUT generation ---
lut = compute(grid_spec, stations, method="1dfmm", vmod=vmod, log=True,
              save_file=lut_file)

# --- NLLoc sweep LUT generation ---
lut = compute(grid_spec, stations, method="1dsweep", vmod=vmod,
              block_model=True, log=True, save_file=lut_file)

# --- Read NLLoc lookup tables ---
lut = read_nlloc("/path/to/nlloc_files", stations, log=True,
                 save_file=lut_file)

# -*- coding: utf-8 -*-

"""
This script will create the traveltime lookup tables for QuakeMigrate

Author: Conor Bacon
"""

from pyproj import Proj

import QMigrate.lut.create_lut as clut
import QMigrate.lut.lut as qlut
import QMigrate.io.quakeio as qio

# Set i/o paths
lut_path = "/path/to/lookup_table"
stat_path = "/path/to/station_file"
vmod_path = "/path/to/vmodel_file"

stations = qio.stations(stat_path)

# Create projection definitions
cproj = Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
gproj = Proj("+proj=lcc +lon_0=116.75 +lat_0=6.25 +lat_1=4.0 +lat_2=7.5 +datum=WGS84 +units=m +no_defs") 

# Create a new instance of LUT
lut = qlut.LUT(ll_corner=[116.075, 5.573, -1750],
               ur_corner=[117.426, 6.925, 27750],
               cell_size=[500., 500., 500.], grid_proj=gproj, coord_proj=cproj)

# --- Homogeneous LUT generation ---
clut.compute(lut, stations, method="homogeneous", vp=5., vs=3.)

# --- skfmm LUT generation ---
clut.compute(lut, stations, method="1dfmm", vmod=vmod_path)

# --- NLLoc sweep LUT generation ---
clut.compute(lut, stations, method="1dsweep", vmod=vmod_path, block_model=True)

# --- Read NLLoc lookup tables ---
nlloc_files = "/path/to/nlloc_files"
lut = clut.read_nlloc(nlloc_files, stations)

# Output LUT
lut.save(lut_path)

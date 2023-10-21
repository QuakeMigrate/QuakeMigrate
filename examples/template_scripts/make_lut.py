# -*- coding: utf-8 -*-
"""
This script demonstrates how to generate traveltime look-up tables (LUT's) for
QuakeMigrate.

"""

from obspy.core import AttribDict
from pyproj import Proj

from quakemigrate.io import read_stations, read_vmodel
from quakemigrate.lut import compute_traveltimes, read_nlloc

# --- i/o paths ---
lut_file = "/path/to/save/lut_file"
station_file = "/path/to/station_file"
vmodel_file = "/path/to/vmodel_file"

# --- Read in the stations and velocity model files ---
stations = read_stations(station_file)
vmod = read_vmodel(vmodel_file)

# --- Define the input and grid projections ---
gproj = Proj(proj="lcc", units="km", lon_0=116.75, lat_0=6.25, lat_1=5.9,
             lat_2=6.6, datum="WGS84", ellps="WGS84", no_defs=True)
cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)

# --- Define the grid specifications ---
# AttribDict behaves like a Python dict, but also has '.'-style access.
grid_spec = AttribDict()
grid_spec.ll_corner = [116.075, 5.573, -1.750]
grid_spec.ur_corner = [117.426, 6.925, 27.750]
grid_spec.node_spacing = [0.5, 0.5, 0.5]
grid_spec.grid_proj = gproj
grid_spec.coord_proj = cproj

# --- Homogeneous LUT generation ---
# P & S
lut = compute_traveltimes(grid_spec, stations, method="homogeneous",
                          phases=["P", "S"], vp=5.0, vs=3.0, log=True,
                          save_file=lut_file)
# Just P
lut = compute_traveltimes(grid_spec, stations, method="homogeneous",
                          phases=["P"], vp=5.0, log=True, save_file=lut_file)

# --- skfmm LUT generation ---
# For P & S you must specify a velocity model with both P and S velocities.
lut = compute_traveltimes(grid_spec, stations, method="1dfmm", vmod=vmod,
                          phases=["P", "S"], log=True, save_file=lut_file)

# --- NonLinLoc LUT generation (using the Grid2Time eikonal solver) ---
# For P & S you must specify a velocity model with both P and S velocities.
lut = compute_traveltimes(grid_spec, stations, method="1dnlloc", vmod=vmod,
                          phases=["P", "S"], block_model=False, log=True,
                          save_file=lut_file)

# --- Read NonLinLoc lookup tables ---
lut = read_nlloc("/path/to/nlloc_files", stations, log=True,
                 save_file=lut_file)

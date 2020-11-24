# -*- coding: utf-8 -*-
"""
This script generates the traveltime look-up table (LUT) for the Iceland dike
intrusion example.

"""

from obspy.core import AttribDict
from pyproj import Proj

from quakemigrate.io import read_stations, read_vmodel
from quakemigrate.lut import compute_traveltimes

# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
vmodel_file = "./inputs/iceland_vmodel.txt"
lut_out = "./outputs/lut/dike_intrusion.LUT"

# --- Read in the station information file ---
stations = read_stations(station_file)

# --- Read in the velocity model file ---
vmodel = read_vmodel(vmodel_file)

# --- Define the input and grid projections ---
gproj = Proj(proj="lcc", units="km", lon_0=-16.9, lat_0=64.8, lat_1=64.7,
             lat_2=64.9, datum="WGS84", ellps="WGS84", no_defs=True)
cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)

# --- Define the grid specifications ---
# AttribDict behaves like a Python dict, but also has '.'-style access.
grid_spec = AttribDict()
grid_spec.ll_corner = [-17.2, 64.7, -2.0]
grid_spec.ur_corner = [-16.6, 64.95, 16.0]
grid_spec.node_spacing = [0.5, 0.5, 0.5]
grid_spec.grid_proj = gproj
grid_spec.coord_proj = cproj

# --- 1-D velocity model LUT generation (using NonLinLoc 1-D Sweep) ---
lut = compute_traveltimes(grid_spec, stations, method="1dsweep", vmod=vmodel,
                          phases=["P", "S"], log=True, save_file=lut_out)

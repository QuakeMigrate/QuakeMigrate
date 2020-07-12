# -*- coding: utf-8 -*-
"""
Creation of LUT for the Rutford icequake example.

"""

from obspy.core import AttribDict
from pyproj import Proj

from QMigrate.io import read_stations
from QMigrate.lut import compute_traveltimes

station_file = "./inputs/rutford_stations.txt"
lut_out = "./outputs/lut/icequake.LUT"

# --- Read in the station information file ---
stations = read_stations(station_file)

# --- Define the input and grid projections ---
gproj = Proj(proj="lcc", units="km", lon_0=-83.932, lat_0=-78.144, lat_1=-78.1,
             lat_2=-77.9, datum="WGS84", ellps="WGS84", no_defs=True)
cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)

# --- Define the grid specifications ---
# AttribDict behaves like a Python dict, but also has '.'-style access.
grid_spec = AttribDict()
grid_spec.ll_corner = [-84.14853353566141, -78.18825429331356, -0.350]
grid_spec.ur_corner = [-83.71921885073093, -78.10003166259442, 3.550]
grid_spec.node_spacing = [0.1, 0.1, 0.1]
grid_spec.grid_proj = gproj
grid_spec.coord_proj = cproj

# --- Homogeneous LUT generation ---
lut = compute_traveltimes(grid_spec, stations, method="homogeneous", vp=3.841,
                          vs=1.970, log=True, save_file=lut_out)

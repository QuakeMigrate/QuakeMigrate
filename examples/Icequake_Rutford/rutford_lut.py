# -*- coding: utf-8 -*-
"""
Creation of LUT for the Rutford icequake example.

"""

from pyproj import Proj

from QMigrate.io import read_stations
from QMigrate.lut import compute_traveltimes

station_file = "./inputs/rutford_stations.txt"
lut_out = "./outputs/lut/icequake.LUT"

# --- Read in the station information file ---
stations = read_stations(station_file)

# --- Define the grid specifications ---
grid_spec = {"ll_corner": [-84.14853353566141, -78.18825429331356, -0.350],
             "ur_corner": [-83.71921885073093, -78.10003166259442, 3.550],
             "cell_size": [0.1, 0.1, 0.1],
             "grid_proj": Proj(proj="lcc", lon_0=-83.932, lat_0=-78.144,
                               lat_1=-78.1, lat_2=-77.9, datum="WGS84",
                               ellps="WGS84", units="km", no_defs=True),
             "coord_proj": Proj(proj="longlat", ellps="WGS84", datum="WGS84",
                                no_defs=True)}

# --- Homogeneous LUT generation ---
lut = compute_traveltimes(grid_spec, stations, method="homogeneous", vp=3.841,
                          vs=1.970, log=True, save_file=lut_out)

# -*- coding: utf-8 -*-
"""
Creation of LUT for the Rutford icequake example.

"""

from pyproj import Proj

import QMigrate.io as qio
import QMigrate.lut as qlut

station_file = "./inputs/rutford_stations.txt"
data_in = "./inputs/mSEED"
lut_out = "./outputs/LUT/icequake.LUT"
out_path = "./outputs/runs"
run_name = "icequake_example"

# --- Read in the station information file ---
stations = qio.stations(station_file)

# --- Define projections ---
cproj = Proj(proj="longlat", ellps="WGS84", datum="WGS84", no_defs=True)
gproj = Proj(proj="lcc", lon_0=-83.932, lat_0=-78.144, lat_1=-78.1, lat_2=-77.9,
             datum="WGS84", ellps="WGS84", units="m", no_defs=True)

# --- Create new LUT ---
# Cell count (x,y,z); cell size (x,y,z in metres)
lut = qlut.LUT(ll_corner=[-84.14853353566141, -78.18825429331356, -350.],
               ur_corner=[-83.71921885073093, -78.10003166259442, 3550],
               cell_size=[100., 100., 100.], grid_proj=gproj, coord_proj=cproj)

# --- Homogeneous LUT generation ---
vp = 3841
vs = 1970
qlut.compute(lut, stations, method="homogeneous", vp=vp, vs=vs)

# --- Save LUT ---
lut.save(lut_out)

"""
This script generates the traveltime lookup table for the synthetic example described in
the tutorial in the online documentation. 

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

# Stop numpy using all available threads (these environment variables must be
# set before numpy is imported for the first time).
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

import numpy as np
from obspy.core import AttribDict
import pandas as pd
from pyproj import Proj
from quakemigrate.io import read_vmodel
from quakemigrate.lut import compute_traveltimes


# Build synthetic lookup table
station_file = "./inputs/synthetic_stations.txt"
vmodel_file = "./inputs/velocity_model.csv"
lut_out = "./outputs/lut/example.LUT"

# --- Build station file ---
rng = np.random.default_rng(13)  # Fix seed for reproducible results
stations = pd.DataFrame()
stations["Network"] = ["SC"] * 10
stations["Name"] = [f"STA{i}" for i in range(10)]
stations["Longitude"] = rng.uniform(low=-0.15, high=0.15, size=10)
stations["Latitude"] = rng.uniform(low=-0.15, high=0.15, size=10)
stations["Elevation"] = rng.uniform(low=-0.0, high=1.0, size=10)
stations.to_csv(station_file, index=False)

# --- Read in the velocity model file ---
vmodel = read_vmodel(vmodel_file)

# --- Define the input and grid projections ---
gproj = Proj(
    proj="lcc",
    units="km",
    lon_0=0.0,
    lat_0=0.0,
    lat_1=-0.10,
    lat_2=0.101,
    datum="WGS84",
    ellps="WGS84",
    no_defs=True,
)
cproj = Proj(proj="longlat", datum="WGS84", ellps="WGS84", no_defs=True)

# --- Define the grid specifications ---
# AttribDict behaves like a Python dict, but also has '.'-style access.
grid_spec = AttribDict()
grid_spec.ll_corner = [-0.15, -0.15, -1.0]
grid_spec.ur_corner = [0.15, 0.15, 30.0]
grid_spec.node_spacing = [0.5, 0.5, 0.5]
grid_spec.grid_proj = gproj
grid_spec.coord_proj = cproj

# --- Homogeneous LUT generation ---
lut = compute_traveltimes(
    grid_spec,
    stations,
    method="1dnlloc",
    vmod=vmodel,
    phases=["P", "S"],
    log=True,
    save_file=lut_out,
)
print()
print(lut)

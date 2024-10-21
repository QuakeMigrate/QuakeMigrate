Building the traveltime lookup table
====================================

For this synthetic example, we use a hypothetical earthquake region situated at (0.0°, 0.0°), spanning ±0.15° in longitude and latitude, and 31 km in depth, spanning from -1 km to 30 km. A randomly distributed network of seismometers are generated, using a fixed random seed, to ensure the same random distribution is generated each time the example is run.

The station file is created by the following section of code:

.. code-block:: python

    station_file = "./inputs/synthetic_stations.txt"

    # --- Build station file ---
    rng = np.random.default_rng(13)  # Fix seed for reproducible results
    stations = pd.DataFrame()
    stations["Network"] = ["SC"] * 10
    stations["Name"] = [f"STA{i}" for i in range(10)]
    stations["Longitude"] = rng.uniform(low=-0.15, high=0.15, size=10)
    stations["Latitude"] = rng.uniform(low=-0.15, high=0.15, size=10)
    stations["Elevation"] = rng.uniform(low=-0.0, high=1.0, size=10)
    stations.to_csv(station_file, index=False)

.. note::

    You can change the seed (or remove it) to explore how the synthetic example performs for different distributions of seismometers in the network.

A simple 1-D velocity model is used, with P- and S-phase velocities for depths spanning a range greater than the targeted search depths. The input velocity model is stored in a ``.csv`` file and read in using a utility function provided by QuakeMigrate that performs some initial sanity checks, e.g., the ``.csv`` must contain at least a ``Depth`` and a ``Vp`` column.

.. code-block:: python

    vmodel_file = "./inputs/velocity_model.csv"

    # --- Read in the velocity model file ---
    vmodel = read_vmodel(vmodel_file)

We employ the Lambert Conformal Conic projection as the basis of our transformation between the geographical coordinate space and a Cartesian coordinate space, on which the grid search is performed. These projections are defined using the :mod:`pyproj` package, which provides an interface to the :mod:`PROJ` library.

.. code-block:: python

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

These projections are combined with a specification of the search grid (per the geographical specifications at the top of this tutorial) using the ``AttribDict`` class, which is an :mod:`obspy` utility class that can be used to construct an object that behaves like a dictionary (i.e., can be accessed using ``dict[key]``) as well as providing ``.``-style access to the (key, value) pairs.

.. code-block:: python

    # --- Define the grid specifications ---
    # AttribDict behaves like a Python dict, but also has '.'-style access.
    grid_spec = AttribDict()
    grid_spec.ll_corner = [-0.15, -0.15, -1.0]
    grid_spec.ur_corner = [0.15, 0.15, 30.0]
    grid_spec.node_spacing = [0.5, 0.5, 0.5]
    grid_spec.grid_proj = gproj
    grid_spec.coord_proj = cproj

Finally, we bring all of these parts together to compute the traveltime grids—that is, for every station, we compute the traveltime from the position of the station to each node in the grid, for each phase (here P and S). In this instance, we make use of the :mod:`NonLinLoc` package to compute these traveltimes. For more information, please refer to the dedicated :doc:`lookup table documentation <../lut>`.

.. code-block:: python

    lut_out = "./outputs/lut/example.LUT"

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

The final line will print out an overview of the travetime lookup table.

The full script looks like this:

.. code-block:: python

    """
    This script generates the traveltime lookup table for the synthetic example described in
    the tutorial in the online documentation. 

    :copyright:
        2020–2024, QuakeMigrate developers.
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

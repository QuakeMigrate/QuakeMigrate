# -*- coding: utf-8 -*-
"""
Module to produce travel-time lookup tables defined on a Cartesian grid.

"""

import os
import pathlib
import struct

import numpy as np
import pandas as pd
import pyproj
from scipy.interpolate import interp1d
import skfmm

import QMigrate.core.lut as qlut


def read_nlloc(path, stations):
    """
    Read in a travel-time lookup table that is saved in the NonLinLoc format.

    Parameters
    ----------
    path : str
        Path to directory containing .buf and .hdr files.

    Returns
    -------
    lut : QuakeMigrate lookup table object
        Lookup table populated with travel times from the NonLinLoc files.

    """

    for i, station in stations.iterrows():
        name = station["Name"]
        msg = "Loading P- and S- travel-time lookup tables for {}".format(name)
        print(msg)

        p_file = path / name
        s_file = path / name

        if i == 1:
            gridspec, transform, pttimes = _read_nlloc(p_file)
            _, _, sttimes = _read_nlloc(s_file)
            cell_count = np.array(gridspec[0])
            grid_origin = np.array(gridspec[1]) * 1000
            cell_size = np.array(gridspec[2]) * 1000

            gproj, cproj = transform
            if gproj is None:
                raise NotImplementedError
            else:
                # Transform from input projection origin to a grid origin
                ll_corner = pyproj.transform(gproj, cproj, grid_origin[0],
                                             grid_origin[1], grid_origin[2])

                # Calculate the ur corner
                ur_corner = np.array(grid_origin) + (cell_count - 1)*cell_size
                ur_corner = pyproj.transform(gproj, cproj, ur_corner[0],
                                             ur_corner[1], ur_corner[2])

            # Need to initialise the grid
            lut = qlut.LUT(ll_corner=ll_corner, ur_corner=ur_corner,
                           cell_size=cell_size, grid_proj=gproj,
                           coord_proj=cproj)
        else:
            _, _, pttimes = _read_nlloc(p_file)
            _, _, sttimes = _read_nlloc(s_file)

        lut.maps[station["Name"]] = {"TIME_P": pttimes,
                                     "TIME_S": sttimes}

    lut.station_data = stations

    return lut


def compute(lut, stations, method, **kwargs):
    """
    Top-level method for computing travel-time lookup tables.

    Parameters
    ----------
    lut : QuakeMigrate lookup table object
        Defines the grid on which the travel times are to be calculated.

    stations : pandas DataFrame
        DataFrame containing station information (lat/lon/elev).

    method : str
        Method to be used when computing the travel-time lookup tables.
            "homogeneous" - straight line velocities
            "1dfmm" - 1-D fast-marching method using scikit-fmm

    kwargs : dict
        Dictionary of all keyword arguments passed to compute when called.
        For lists of valid arguments, please refer to the relevant method.

    """

    stations["Elevation"] = stations["Elevation"].apply(lambda x: -1*x)
    lut.station_data = stations

    if method == "homogeneous":
        # Check the user has provided the suitable arguments
        if "vp" not in kwargs:
            print("Missing argument: 'vp'")
            return
        if "vs" not in kwargs:
            print("Missing argument: 'vs'")
            return

        _compute_homogeneous(lut, kwargs["vp"], kwargs["vs"])

    if method == "1dfmm":
        # Check if the user has provided suitable arguments
        if "vmod" in kwargs:
            vmod = kwargs["vmod"].values
        elif "vmod_file" in kwargs:
            if "header" in kwargs:
                header = kwargs["header"]
            else:
                header = None
            if "delimiter" in kwargs:
                delimiter = kwargs["delimiter"]
            else:
                delimiter = ","
            vmod = pd.read_csv(kwargs["vmod_file"], header=header,
                               delimiter=delimiter).values

        _compute_1d_fmm(lut, vmod)

    if method == "3dfmm":

        raise NotImplementedError

    if method == "1dsweep":
        # Check if the user has provided suitable arguments
        if "vmod" in kwargs:
            vmod = kwargs["vmod"].values
        elif "vmod_file" in kwargs:
            if "header" in kwargs:
                header = kwargs["header"]
            else:
                header = None
            if "delimiter" in kwargs:
                delimiter = kwargs["delimiter"]
            else:
                delimiter = ","
            vmod = pd.read_csv(kwargs["vmod_file"], header=header,
                               delimiter=delimiter).values

        if "blocks" not in kwargs:
            blocks = False

        _compute_1d_sweep(lut, vmod, blocks)

    return lut


def _compute_homogeneous(lut, vp, vs):
    """
    Calculate the travel-time lookup table for a station in a homogeneous
    velocity model.

    Parameters
    ----------
    lut : QuakeMigrate lookup table object
        Defines the grid on which the travel times are to be calculated.

    vp : float
        P-wave velocity (units: km / s)

    vs : float
        S-wave velocity (units: km / s)

    """

    grid_xyz = lut.grid_xyz
    stations_xyz = lut.stations_xyz

    for i, station in lut.station_data.iterrows():
        msg = "Computing homogeneous travel-time lookup table - {} of {}"
        msg = msg.format(i + 1, stations_xyz.shape[0])
        print(msg)

        dx, dy, dz = [grid_xyz[j] - stations_xyz[i, j] for j in range(3)]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        lut.maps[station["Name"]] = {"TIME_P": dist / vp,
                                     "TIME_S": dist / vs}


def _compute_1d_fmm(lut, vmod):
    """
    Calculate travel-time lookup tables for each station in a 1-D velocity
    model using the fast-marching method.

    Parameters
    ----------
    lut : QuakeMigrate lookup table object
        Defines the grid on which the travel times are to be calculated.

    vmod : array-like
        Array containing the velocity model to be used to generate the LUT.
        Columns: ["Z", "Vp", "Vs"]
            Z : Depth of each layer in model (positive up; units: metres)
            Vp : P-wave velocity for each layer in model (units: km / s)
            Vs : S-wave velocity for each layer in model (units: km / s)

    """

    z, vp, vs = vmod[:, 0], vmod[:, 1] * 1000, vmod[:, 2] * 1000

    finfo = np.finfo(float)
    z = np.insert(np.append(z, finfo.max), 0, finfo.min)
    vp = np.insert(np.append(vp, vp[-1]), 0, vp[0])
    vs = np.insert(np.append(vs, vs[-1]), 0, vs[0])

    grid_xyz = lut.grid_xyz
    stations_xyz = lut.stations_xyz

    # Interpolate the velocity model in the Z-dimension
    f = interp1d(z, vp)
    gvp = f(grid_xyz[2])
    f = interp1d(z, vs)
    gvs = f(grid_xyz[2])

    for i, station in lut.station_data.iterrows():
        msg = "Computing 1-D fast-marching travel-time lookup table - {} of {}"
        msg = msg.format(i + 1, stations_xyz.shape[0])
        print(msg)

        lut.maps[station["Name"]] = {"TIME_P": _eikonal(grid_xyz,
                                                        lut.cell_size, gvp,
                                                        stations_xyz[i]),
                                     "TIME_S": _eikonal(grid_xyz,
                                                        lut.cell_size, gvs,
                                                        stations_xyz[i])}


def _eikonal(grid_xyz, cell_size, velocity_grid, station_xyz):
    """
    Calculates the travel-time lookup tables by solving the eikonal equation
    using an implementation of the fast-marching algorithm.

    Requires the skifmm python package.

    Parameters
    ----------
    grid_xyz : array-like
        [X, Y, Z] coordinates of each cell

    cell_size : array-like
        [X, Y, Z] dimensions of each cell

    velocity_grid : array-like
        Contains the speed of interface propagation at each point in the
        domain

    station_xyz : array-like
        Station location (in grid xyz)

    Returns
    -------
    t : array-like, same shape as phi
        Contains the travel time from the zero contour (zero level set) of phi
        to each point in the array given the scalar velocity field speed. If
        the input array speed has values less than or equal to zero the return
        value will be a masked array.

    """

    phi = -np.ones(grid_xyz[0].shape)
    indx = np.argmin(abs(grid_xyz[0] - station_xyz[0])
                     + abs(grid_xyz[1] - station_xyz[1])
                     + abs(grid_xyz[2] - station_xyz[2]))
    phi[np.unravel_index(indx, grid_xyz[0].shape)] = 1.0

    t = skfmm.travel_time(phi, velocity_grid, dx=cell_size)
    return t


def _compute_1d_sweep(lut, vmod, nlloc_dx=0.1, nlloc_path="", blocks=False):
    """
    Calculate 3-D travel-time lookup-tables from a 1-D velocity model.

    NonLinLoc Grid2Time is used to generate a 2-D lookup-table which is then
    swept across a 3-D distance from station grid to populate a 3-D travel-time
    grid.

    Parameters
    ----------
    lut : QuakeMigrate lookup table object
        Defines the grid on which the travel times are to be calculated.

    vmod : array-like
        Array containing the velocity model to be used to generate the LUT.
        Columns: ["Z", "Vp", "Vs"]
            Z : Depth of each layer in model (positive up; units: metres)
            Vp : P-wave velocity for each layer in model (units: km / s)
            Vs : S-wave velocity for each layer in model (units: km / s)

    nlloc_dx : float, optional
        NLLoc 2D grid spacing (default: 0.1 km).

    nlloc_path : str
        Path to NonLinLoc binaries.

    blocks : bool
        Toggle to choose whether to interpret velocity model with constant
        velocity blocks or a linear gradient.

    """

    from subprocess import check_output, STDOUT

    lut.velocity_model = vmod
    cc = lut.cell_count

    grid_xyz = lut.grid_xyz
    stations_xyz = lut.stations_xyz

    # Make folders in which to run NonLinLoc
    pathlib.Path.cwd().mkdir("time", exist_ok=True)
    pathlib.Path.cwd().mkdir("model", exist_ok=True)

    for i, station in lut.station_data.iterrows():
        msg = "Computing 1-D sweep travel-time lookup table - {} of {}"
        msg = msg.format(i + 1, stations_xyz.shape[0])
        print(msg)

        name = station["Name"]
        lut.maps[name] = {}

        # For NonLinLoc, distances must be in km
        dx, dy, dz = [grid_xyz[j] - stations_xyz[i, j] for j in range(3)]
        distance_grid = np.sqrt(dx**2 + dy**2)
        max_dist = np.max(distance_grid)

        # NLLoc needs the station to lie within the 2D section -> we pick the
        # depth extent of the 2-D grid from the max. possible extent of the
        # station and the grid
        min_z = np.min([lut.grid_corners[0][2], stations_xyz[i, 2]])
        max_z = np.max([lut.grid_corners[-1][2], stations_xyz[i, 2]])
        depth_extent = np.array([min_z, max_z])

        for phase in ["P", "S"]:
            # Allow 2 nodes on depth extent as a computational buffer
            _write_control_file(stations_xyz[i], name, max_dist, vmodel,
                                depth_extent, phase=phase, dx=nlloc_dx,
                                block_model=blocks)

            print("\tRunning NonLinLoc - phase =", phase)
            for mode in ["Vel2Grid", "Grid2Time"]:
                out = check_output([os.path.join(nlloc_path, mode),
                                    "control.in"], stderr=STDOUT)
                if b"ERROR" in out:
                    raise Exception("{} Error".format(mode), out)

            to_read = "./time/layer.{}.{}.time".format(phase, name)
            gridspec, _, ttimes = _read_nlloc(to_read)

            distance = distance_grid.flatten()
            depth = grid_xyz[2].flatten()
            ttimes = _bilinear_interpolate(np.c_[distance, depth],
                                           gridspec[:, 1:], ttimes[0, :, :])

            lut.maps[name]["TIME_{}".format(phase)] = ttimes.reshape(cc)


def _write_control_file(station_xyz, name, max_dist, vmodel, depth_limits,
                        phase, dx, blocks):
    """
    Write out a control file for NonLinLoc.

    Parameters
    ----------
    station_xyz : array-like
        Station location expressed in the coordinate space of the grid.

    name : str
        Station name.

    max_dist : float
        Maximum distance between the station and any point in the grid.

    vmodel : pandas DataFrame
        Contains columns with names "depth", "vp" and "vs".

    depth_limits : array-like
        Minimum/maximum extent of the grid in the z-dimension.

    phase : str
        Seismic phase to assign to the velocity model.

    dx : float
        NLLoc 2D grid spacing (default: 0.1 km).

    blocks : bool
        Toggle to choose whether to interpret velocity model with constant
        velocity blocks or a linear gradient.

    """

    control_string = ("CONTROL 0 54321\n",
                      "TRANS NONE\n\n",
                      "VGOUT ./model/layer\n",
                      "VGTYPE {phase:s}\n\n",
                      "VGRID {grid:s} SLOW_LEN\n\n",
                      "{vmodel:s}\n\n",
                      "GTFILES ./model/layer ./time/layer {phase:s}\n",
                      "GTMODE GRID2D ANGLES_NO\n\n",
                      "GTSRCE {name:s} XYZ {x:f} {y:f} {z:f} 0.0\n\n",
                      "GT_PLFD 1.0E-3 0")

    out = control_string.format(phase=phase,
                                grid=_grid_string(max_dist, depth_limits, dx),
                                vmodel=_vmodel_string(vmodel, blocks),
                                name=name, x=station_xyz[0], y=station_xyz[1],
                                z=station_xyz[2])

    with open("./control.in", "w") as f:
        f.write(out)


def _read_nlloc(fname):
    """
    Read the 2-D NonLinLoc travel-time grids

    Parameters
    ----------
    fname : str
        Path to file containing NonLinLoc travel-time lookup tables, without
        the extension.

    Returns
    -------
    ttimes : array-like
        Travel-times for the station.

    gridspec : array-like
        Details on the NonLinLoc grid specification. Contains the number of
        cells, the grid origin and the cell dimensions.

    """

    with open(fname + ".hdr", "r") as f:
        # Read the grid definition line
        line = f.readline().split()
        nx, ny, nz = int(line[0]), int(line[1]), int(line[2])
        x0, y0, z0 = float(line[3]), float(line[4]), float(line[5])
        dx, dy, dz = float(line[6]), float(line[7]), float(line[8])

        # Read the station information line
        _ = f.readline().split()

        # Read the transform definition line
        line = f.readline().split()
        cproj = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
        gproj = None
        if line[1] == "NONE":
            print("No projection selected.")
        elif line[1] == "SIMPLE":
            orig_lat = float(line[3])
            orig_lon = float(line[5])

            gproj = pyproj.Proj(proj='eqc', lat_0=orig_lat, lon_0=orig_lon)
        elif line[1] == "LAMBERT":
            proj_ellipsoid = line[3]
            orig_lat = float(line[5])
            orig_lon = float(line[7])
            parallel_1 = float(line[9])
            parallel_2 = float(line[11])

            gproj = pyproj.Proj(proj="LCC", lon0=orig_lon, lat0=orig_lat,
                                parallel_1=parallel_1, parallel_2=parallel_2)
        elif line[1] == "TRANS_MERC":
            orig_lat = float(line[5])
            orig_lon = float(line[7])

            gproj = pyproj.Proj(proj="TM", lon=orig_lon, lat=orig_lat)

        transform = [gproj, cproj]

    with open("{}.buf".format(fname), "rb") as f:
        npts = nx * ny * nz
        buf = f.read(npts * 4)
        ttimes = struct.unpack("f" * npts, buf)

    ttimes = np.array(ttimes).reshape((nx, ny, nz))
    gridspec = np.array([[nx, ny, nz], [x0, y0, z0], [dx, dy, dz]])

    return gridspec, transform, ttimes


def _bilinear_interpolate(xz, xz_origin, xz_dimensions, ttimes):
    """
    Perform a bi-linear interpolation between 4 data points on the input
    2-D lookup table to calculate the travel time to nodes on the 3-D grid.

    Note: x is used to denote the horizontal dimension over which the
    interpolation is performed. Due to the NonLinLoc definition, this
    corresponds to the y component of the grid.

    Parameters
    ----------
    xz : array-like
        Column-stacked array of distances from the station and depths for all
        points in grid.

    xz_origin : array-like
        The x (actually y) and z values of the grid origin.

    xz_dimensions : array-like
        The x (actually y) and z values of the cell dimensions.

    ttimes : array-like
        A slice through the travel-time grid at x = 0, on which to perform the
        interpolation.

    Returns
    -------
    c : array-like
        Interpolated 3-D travel-time lookup table.

    """

    # Get indices of the nearest node
    i, k = np.floor((xz - xz_origin) / xz_dimensions).astype(int).T

    # Get fractional distance along each axis
    x_d, z_d = np.remainder(xz, xz_dimensions) / xz_dimensions

    # Get 4 data points of surrounding square
    c00 = ttimes[i, k]
    c10 = ttimes[i+1, k]
    c11 = ttimes[i+1, k+1]
    c01 = ttimes[i, k+1]

    # Interpolate along x-axis
    c0 = c00 * (1 - x_d) + c10 * x_d
    c1 = c01 * (1 - x_d) + c11 * x_d

    # Interpolate along z-axis (c = c0 if np.all(z_d == 0.))
    c = c0 * (1 - z_d) + c1 * z_d

    return c


def _vmodel_string(vmodel, blocks):
    """
    Creates a string representation of the velocity model for the control file.

    Parameters
    ----------
    vmodel : pandas DataFrame
        Contains columns with names "depth", "vp" and "vs".

    blocks : bool
        Toggle to choose whether to interpret velocity model with constant
        velocity blocks or a linear gradient.

    Returns
    -------
    str_vmodel : str
        NonLinLoc formatted string describing the velocity model.

    """

    string_template = "LAYER  {0:f} {1:f} {3:f} {2:f} {4:f} 0.0 0.0"

    out = []

    i = 0
    while i < len(vmodel):
        if not blocks:
            try:
                dvp, dvs = _velocity_gradient(i, vmodel)
            except KeyError:
                dvp = dvs = 0.0
        else:
            dvp = dvs = 0.0

        out.append(string_template.format(vmodel["depth"][i] / 1000.,
                                          vmodel["vp"][i] / 1000.,
                                          vmodel["vs"][i] / 1000.,
                                          dvp, dvs))
        i += 1

    return "\n".join(out)


def _velocity_gradient(i, vmodel):
    """
    Calculate the gradient of the velocity model between two layers.

    Parameters
    ----------
    i : int
        Index of upper layer.

    vmodel : pandas DataFrame
        Contains columns with names "depth", "vp" and "vs".

    Returns
    -------
    dvp, dvs : float
        P- and S-velocity gradients.

    """

    d_depth = vmodel["depth"][i+1] - vmodel["depth"][i]
    d_vp = vmodel["vp"][i+1] - vmodel["vp"][i]
    d_vs = vmodel["vs"][i+1] - vmodel["vs"][i]

    return d_vp / d_depth, d_vs / d_depth


def _grid_string(max_dist, depth_limits, dx):
    """
    Creates a string representation of the grid for the control file.

    Parameters
    ----------
    max_dist : float
        Maximum distance between the station and any point in the grid.

    depth_limits : array-like
        Minimum/maximum extent of the grid in the z-dimension.

    dx : float
        NLLoc 2D grid spacing (default: 0.1 km).

    Returns
    -------
    str_grid : str
        NonLinLoc formatted string describing the grid.

    """

    string_template = "2 {0:d} {1:d} 0.0 0.0 {2:f} {3:f} {3:f} {3:f}"

    max_x = int(np.ceil(max_dist / dx)) + 5
    max_z = int(np.ceil((depth_limits[1] - depth_limits[0]) / dx)) + 5

    return string_template.format(max_x, max_z, depth_limits[0], dx)

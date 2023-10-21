# -*- coding: utf-8 -*-
"""
Module to produce traveltime lookup tables defined on a Cartesian grid.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging
import warnings
import os
import pathlib
import struct
from shutil import rmtree

import numpy as np
from pyproj import Proj, Transformer
from scipy.interpolate import interp1d

import quakemigrate.util as util
from .lut import LUT


def read_nlloc(
    path, stations, phases=["P", "S"], fraction_tt=0.1, save_file=None, log=False
):
    """
    Read in a traveltime lookup table that is saved in the NonLinLoc format.

    Parameters
    ----------
    path : str
        Path to directory containing .buf and .hdr files.
    stations : `pandas.DataFrame`
        DataFrame containing station information (lat/lon/elev).
    phases : list of str, optional
        List of seismic phases for which to read in traveltimes.
    fraction_tt : float, optional
        An estimate of the uncertainty in the velocity model as a function of
        a fraction of the traveltime. (Default 0.1 == 10%)
    save_file : str, optional
        Path to location to save pickled lookup table.
    log : bool, optional
        Toggle for logging - default is to only print information to stdout.
        If True, will also create a log file.

    Returns
    -------
    lut : :class:`~quakemigrate.lut.lut.LUT` object
        Lookup table populated with traveltimes from the NonLinLoc lookup table files.

    Raises
    ------
    NotImplementedError
        If the specified projection type is not supported.

    """

    path = pathlib.Path(path)
    util.logger(pathlib.Path.cwd() / "logs" / "lut", log)

    logging.info("Loading NonLinLoc traveltime lookup tables for...")
    for i, phase in enumerate(phases):
        logging.info(f"\t...phase: {phase}...")
        for j, station in enumerate(stations["Name"].values):
            logging.info(f"\t\t...station: {station}")
            file = path / f"layer.{phase}.{station}.time"

            if i == 0 and j == 0:
                gridspec, transform, traveltimes = _read_nlloc(file)
                node_count = np.array(gridspec[0])
                grid_origin = np.array(gridspec[1])
                node_spacing = np.array(gridspec[2])

                gproj, cproj, gproj_string = transform
                if gproj is None:
                    raise NotImplementedError(
                        f"Projection type {gproj_string} not supported."
                    )

                # Transform from grid projection origin to a coord origin
                ll_corner = Transformer.from_proj(gproj, cproj).transform(*grid_origin)

                # Calculate the ur corner
                ur_corner = np.array(grid_origin) + (node_count - 1) * node_spacing
                ur_corner = Transformer.from_proj(gproj, cproj).transform(*ur_corner)

                # Need to initialise the grid
                lut = LUT(
                    ll_corner=ll_corner,
                    ur_corner=ur_corner,
                    node_spacing=node_spacing,
                    grid_proj=gproj,
                    coord_proj=cproj,
                    fraction_tt=fraction_tt,
                )
            else:
                _, _, traveltimes = _read_nlloc(file)

            lut.traveltimes.setdefault(station, {}).update({phase: traveltimes})

    lut.station_data = stations
    lut.phases = phases

    if save_file is not None:
        lut.save(save_file)

    return lut


def compute_traveltimes(
    grid_spec,
    stations,
    method,
    phases=["P", "S"],
    fraction_tt=0.1,
    save_file=None,
    log=False,
    **kwargs,
):
    """
    Top-level method for computing traveltime lookup tables.

    This function takes a grid specification and is capable of computing traveltimes for
    an arbitrary number of phases using a variety of techniques.

    Parameters
    ----------
    grid_spec : dict
        Dictionary containing all of the defining parameters for the underlying 3-D grid
        on which the traveltimes are to be calculated. For expected keys, see
        :class:`~quakemigrate.lut.lut.Grid3D`.
    stations : `pandas.DataFrame`
        DataFrame containing station information (lat/lon/elev).
    method : str
        Method to be used when computing the traveltime lookup tables.\n
            "homogeneous" - straight line velocities.\n
            "1dfmm" - 1-D fast-marching method using scikit-fmm.\n
            "1dnlloc" - a 2-D traveltime grid is calculated from the 1-D\
                        velocity model using the Grid2Time eikonal solver in\
                        NonLinLoc, then swept over the 3-D grid using a\
                        bilinear interpolation scheme.
    phases : list of str, optional
        List of seismic phases for which to calculate traveltimes.
    fraction_tt : float, optional
        An estimate of the uncertainty in the velocity model as a function of a fraction
        of the traveltime. (Default 0.1 == 10%)
    save_file : str, optional
        Path to location to save pickled lookup table.
    log : bool, optional
        Toggle for logging - default is to only print information to stdout.
        If True, will also create a log file.
    kwargs : dict
        Dictionary of all keyword arguments passed to compute when called.
        For lists of valid arguments, please refer to the relevant method.

    Returns
    -------
    lut : :class:`~quakemigrate.lut.lut.LUT` object
        Lookup table populated with traveltimes.

    Raises
    ------
    ValueError
        If the specified `method` is not a valid option.
    TypeError
        If the velocity model, or constant phase velocity, is not specified.
    NotImplementedError
        If the `3dfmm` method is specified.

    """

    util.logger(pathlib.Path.cwd() / "logs" / "lut", log)

    lut = LUT(**grid_spec, fraction_tt=fraction_tt)
    lut.station_data = stations
    lut.phases = phases

    if method == "1dsweep":
        warnings.warn(
            "Parameter name has changed - continuing. To remove this"
            " message, change:\t'1dsweep' -> '1dnlloc'.",
            DeprecationWarning,
            2,
        )
        method = "1dnlloc"

    if method == "homogeneous":
        logging.info("Computing homogeneous traveltimes for...")
        lut.velocity_model = "Homogeneous velocity model:"
        for phase in phases:
            velocity = kwargs.get(f"v{phase.lower()}")
            if velocity is None:
                raise TypeError(f"Missing argument: 'v{phase.lower()}'")
            lut.velocity_model += f"\n\tV{phase.lower()} = {velocity:5.2f} m/s"

            logging.info(f"\t...phase: {phase}...")
            _compute_homogeneous(lut, phase, velocity)

    elif method == "1dfmm":
        logging.info("Computing 1-D fast-marching traveltimes for...")
        lut.velocity_model = vmodel = kwargs.get("vmod")
        if vmodel is None:
            raise TypeError("Missing argument: 'vmod'")

        for phase in phases:
            logging.info(f"\t...phase: {phase}...")
            _compute_1d_fmm(lut, phase, vmodel)

    elif method == "3dfmm":
        raise NotImplementedError(
            "Feature coming soon - please contact the QuakeMigrate developers."
        )

    elif method == "1dnlloc":
        logging.info("Computing 1-D nlloc traveltimes for...")
        lut.velocity_model = vmodel = kwargs.get("vmod")
        if vmodel is None:
            raise TypeError("Missing argument: 'vmod'")

        for phase in phases:
            logging.info(f"\t...phase: {phase}...")
            _compute_1d_nlloc(lut, phase, vmodel, **kwargs)

    else:
        raise ValueError(
            f"'{method} is not a valid method. Please consult the documentation. Valid "
            "options are 'homogeneous', '1dfmm', and '1dnlloc'."
        )

    if save_file is not None:
        lut.save(save_file)

    return lut


def _compute_homogeneous(lut, phase, velocity):
    """
    Calculate the traveltime lookup table for a station in a homogeneous velocity model.

    Parameters
    ----------
    lut : :class:`~quakemigrate.lut.lut.LUT` object
        Defines the grid on which the traveltimes are to be calculated.
    phase : str
        The seismic phase for which to calculate traveltimes.
    velocity : float
        Seismic phase velocity.

    """

    grid_xyz = lut.grid_xyz
    stations_xyz = lut.stations_xyz

    for i, station in enumerate(lut.station_data["Name"].values):
        logging.info(f"\t\t...station: {station} - {i+1} of {stations_xyz.shape[0]}")

        dx, dy, dz = [grid_xyz[j] - stations_xyz[i, j] for j in range(3)]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        lut.traveltimes.setdefault(station, {}).update({phase: dist / velocity})


def _compute_1d_fmm(lut, phase, vmodel):
    """
    Calculate traveltime lookup tables for each station in a 1-D velocity model using
    the fast-marching method.

    Parameters
    ----------
    lut : :class:`~quakemigrate.lut.lut.LUT` object
        Defines the grid on which the traveltimes are to be calculated.
    phase : str
        The seismic phase for which to calculate traveltimes.
    vmodel : `pandas.DataFrame` object
        DataFrame containing the velocity model to be used to generate the LUT.
        Columns:
            "Depth" of each layer in model (positive down)
            "V<phase>" velocity for each layer in model (e.g. "Vp")

    Raises
    ------
    InvalidVelocityModelHeader
        If the velocity model does not contain the key corresponding to the specified
        seismic `phase`. (E.g. "Vp" for "P" phase.)

    """

    try:
        depths, vmodel = vmodel[["Depth", f"V{phase.lower()}"]].values.T
    except KeyError:
        raise util.InvalidVelocityModelHeader(f"V{phase.lower()}")

    finfo = np.finfo(float)
    depths = np.insert(np.append(depths, finfo.max), 0, finfo.min)
    vmodel = np.insert(np.append(vmodel, vmodel[-1]), 0, vmodel[0])

    grid_xyz = lut.grid_xyz
    stations_xyz = lut.stations_xyz

    # Check that all stations are contained within grid
    if (lut.stations_xyz < lut.ll_corner).any() or (
        lut.stations_xyz > lut.ur_corner
    ).any():
        raise ValueError(
            "Cannot calculate traveltimes with method '1dfmm' unless all stations are "
            "contained within the grid! Please either use method '1dnlloc' or increase "
            "the grid extent to contain all stations"
        )

    # Interpolate the velocity model in the Z-dimension
    f = interp1d(depths, vmodel)
    int_vmodel = f(grid_xyz[2])

    for i, station in enumerate(lut.station_data["Name"].values):
        logging.info(f"\t\t...station: {station} - {i+1} of {stations_xyz.shape[0]}")

        lut.traveltimes.setdefault(station, {}).update(
            {
                phase: _eikonal_fmm(
                    grid_xyz, lut.node_spacing, int_vmodel, stations_xyz[i]
                )
            }
        )


def _eikonal_fmm(grid_xyz, node_spacing, velocity_grid, station_xyz):
    """
    Calculates the traveltime lookup tables by solving the eikonal equation using an
    implementation of the fast-marching algorithm.

    Traveltime calculation can only be performed between grid nodes: the station
    location is therefore taken as the closest grid node. Note that for large node
    spacings this may cause a modest error in the calculated traveltimes.

    .. warning:: Requires the scikit-fmm python package.

    Parameters
    ----------
    grid_xyz : array-like
        [X, Y, Z] coordinates of each node.
    node_spacing : array-like
        [X, Y, Z] distances between each node.
    velocity_grid : array-like
        Contains the speed of interface propagation at each point in the domain.
    station_xyz : array-like
        Station location (in grid xyz).

    Returns
    -------
    traveltimes : array-like, same shape as grid_xyz
        Contains the traveltime from the zero contour (zero level set) of phi to each
        point in the array given the scalar velocity field speed. If the input array
        speed has values less than or equal to zero the return value will be a masked
        array.

    Raises
    ------
    ImportError
        If scikit-fmm is not installed.

    """

    try:
        import skfmm
    except ImportError:
        raise ImportError(
            "Unable to import skfmm - you need to install scikit-fmm to use this "
            "method.\nSee the installation instructions in the documentation for more "
            "details."
        )

    phi = -np.ones(grid_xyz[0].shape)
    # Find closest grid node to true station location
    indx = np.argmin(
        abs(grid_xyz[0] - station_xyz[0])
        + abs(grid_xyz[1] - station_xyz[1])
        + abs(grid_xyz[2] - station_xyz[2])
    )
    phi[np.unravel_index(indx, grid_xyz[0].shape)] = 1.0

    return skfmm.travel_time(phi, velocity_grid, dx=node_spacing)


def _compute_1d_nlloc(lut, phase, vmodel, **kwargs):
    """
    Calculate 3-D traveltime lookup tables from a 1-D velocity model.

    NonLinLoc Grid2Time is used to generate a 2-D lookup table which is then swept
    around the full range of azimuths, centred on the station location, to populate the
    3-D traveltime grid.

    .. warning:: Requires NonLinLoc to be installed, and `Vel2Grid` and `Grid2Time` to
    be in the user's path. Alternatively, a custom path to these executables can be
    specified with the kwarg `nlloc_path`.

    Parameters
    ----------
    lut : :class:`~quakemigrate.lut.lut.LUT` object
        Defines the grid on which the traveltimes are to be calculated.
    phase : str
        The seismic phase for which to calculate traveltimes.
    vmodel : `pandas.DataFrame` object
        DataFrame containing the velocity model to be used to generate the LUT.
        Columns:
            "Depth" of each layer in model (positive down)
            "V<phase>" velocity for each layer in model (e.g. "Vp")
    kwargs : dict
        Can contain:
        nlloc_dx : float, optional
            NLLoc 2D grid spacing (default: 0.1 km). Note: units must be km.
        nlloc_path : str, optional
            Path to NonLinLoc executables Vel2Grid and Grid2Time (default: "").
        block_model : bool, optional
            Toggle to choose whether to interpret velocity model with constant velocity
            blocks or a linear gradient (default: False).
        retain_nll_grids : bool, optional
            Toggle to choose whether to keep the 2-D traveltime grids created by
            NonLinLoc Grid2Time (default: False).

    Raises
    ------
    FileNotFoundError
        If the Vel2Grid and/or Grid2Time executables are not found in the `nlloc_path`.
    Exception
        If the execution of `Grid2Time` or `Vel2Grid` returns an error.

    """

    from subprocess import check_output, STDOUT

    # Unpack kwargs
    nlloc_dx = kwargs.get("nlloc_dx", 0.1)
    nlloc_path = pathlib.Path(kwargs.get("nlloc_path", ""))
    block_model = kwargs.get("block_model", False)
    retain_nll_grids = kwargs.get("retain_nll_grids", False)

    # Check nlloc_path is valid and contains Vel2Grid and Grid2Time
    if kwargs.get("nlloc_path", "") != "":
        if (
            not (nlloc_path / "Vel2Grid").exists()
            or not (nlloc_path / "Grid2Time").exists()
        ):
            raise FileNotFoundError(
                "Incorrect nlloc_path? - Grid2Time and Vel2Grid not found in "
                f"{nlloc_path}"
            )

    # For NonLinLoc, distances/velocities must be in km - use conversion factor
    km_cf = 1000 / lut.unit_conversion_factor
    grid_xyz = [g / km_cf for g in lut.grid_xyz]
    stations_xyz = lut.stations_xyz / km_cf
    ll, *_, ur = lut.grid_corners / km_cf
    vmodel = vmodel / km_cf

    # Make folders in which to run NonLinLoc
    cwd = pathlib.Path.cwd()
    (cwd / "time").mkdir(exist_ok=True)
    (cwd / "model").mkdir(exist_ok=True)

    for i, station in enumerate(lut.station_data["Name"].values):
        logging.info(
            f"\t\t...running Grid2Time - station: {station:5s} - {i+1} of "
            f"{stations_xyz.shape[0]}"
        )

        dx, dy = [grid_xyz[j] - stations_xyz[i, j] for j in range(2)]
        distances = np.sqrt(dx**2 + dy**2).flatten()
        depths = grid_xyz[2].flatten()
        max_dist = np.max(distances)

        # NLLoc needs the station to lie within the 2-D section -> we pick the depth
        # extent of the 2-D grid from the max. possible extent of the station and the
        # grid - [min_z, max_z]
        depth_span = [
            np.min([ll[2], stations_xyz[i, 2]]),
            np.max([ur[2], stations_xyz[i, 2]]),
        ]

        # Allow 2 nodes on depth extent as a computational buffer
        _write_control_file(
            stations_xyz[i],
            station,
            max_dist,
            vmodel,
            depth_span,
            phase,
            nlloc_dx,
            block_model,
        )

        for mode in ["Vel2Grid", "Grid2Time"]:
            out = check_output([str(nlloc_path / mode), "control.in"], stderr=STDOUT)
            if b"ERROR" in out:
                raise Exception(f"{mode} Error", out)

        to_read = cwd / "time" / f"layer.{phase}.{station}.time"
        gridspec, _, traveltimes = _read_nlloc(to_read, ignore_proj=True)

        lut.traveltimes.setdefault(station, {}).update(
            {
                phase: _bilinear_interpolate(
                    np.c_[distances, depths],
                    gridspec[1, 1:],
                    gridspec[2, 1:],
                    traveltimes[0, :, :],
                ).reshape(lut.node_count)
            }
        )

        # Tidy up: remove control file and nll model and time files
        os.remove(cwd / "control.in")
        if not retain_nll_grids:
            for file in (cwd / "time").glob(f"layer.{phase}.{station}.time*"):
                file.unlink()
            for file in (cwd / "time").glob(f"layer.{phase}.mod.*"):
                file.unlink()

    if not retain_nll_grids:
        rmtree(cwd / "model")
        # Check time directory is empty before removing (might have saved grids from
        # previous runs)
        if not os.listdir(cwd / "time"):
            rmtree(cwd / "time")
        else:
            logging.info(
                "Warning: time directory not empty; does it contain traveltime grids "
                "from a previous run?\nNot removed."
            )


def _write_control_file(
    station_xyz, station, max_dist, vmodel, depth_span, phase, dx, block_model
):
    """
    Write out a control file for NonLinLoc.

    Parameters
    ----------
    station_xyz : array-like
        Station location expressed in the coordinate space of the grid, in km.
    station : str
        Station name.
    max_dist : float
        Maximum distance between the station and any point in the grid, in km.
    vmodel : `pandas.DataFrame` object
        DataFrame containing the velocity model to be used to generate the LUT.
        Columns:
            "Depth" of each layer in model (positive down), in km.
            "V<phase>" velocity for each layer in model (e.g. "Vp"), in km / s.
    depth_span : array-like
        Minimum/maximum extent of the grid in the z-dimension, in km.
    phase : str
        The seismic phase for which to calculate traveltimes.
    dx : float
        NLLoc 2D grid spacing, in km.
    block_model : bool
        Toggle to choose whether to interpret velocity model with constant velocity
        blocks or a linear gradient.

    """

    control_string = (
        "CONTROL 0 54321\n"
        "TRANS NONE\n\n"
        "VGOUT {model_path:s}\n"
        "VGTYPE {phase:s}\n\n"
        "VGGRID {grid:s} SLOW_LEN\n\n"
        "{vmodel:s}\n\n"
        "GTFILES {model_path:s} {time_path:s} {phase:s}\n"
        "GTMODE GRID2D ANGLES_NO\n\n"
        "GTSRCE {station:s} XYZ {x:f} {y:f} {z:f} 0.0\n\n"
        "GT_PLFD 1.0E-3 0"
    )

    cwd = pathlib.Path.cwd()
    out = control_string.format(
        phase=phase,
        grid=_grid_string(max_dist, depth_span, dx),
        vmodel=_vmodel_string(vmodel, block_model, phase),
        model_path=str(cwd / "model" / "layer"),
        time_path=str(cwd / "time" / "layer"),
        station=station,
        x=station_xyz[0],
        y=station_xyz[1],
        z=station_xyz[2],
    )

    with open(cwd / "control.in", "w") as f:
        f.write(out)


def _read_nlloc(fname, ignore_proj=False):
    """
    Read traveltime lookup tables saved in the NonLinLoc format.

    Parses in the header of a NonLinLoc file for the grid specification and the
    coordinate transforms. Then unpacks the binary buffer file containing the
    traveltimes.

    Parameters
    ----------
    fname : str
        Path to file containing NonLinLoc traveltime lookup tables, without the
        extension.
    ignore_proj : bool, optional
        Flag to suppress the "No projection specified" message.

    Returns
    -------
    gridspec : array-like
        Details on the NonLinLoc grid specification. Contains the number of nodes, the
        grid origin and the node spacings.
    transform : array of `pyproj.Proj` objects
        Array containing the grid and coordinate projections, respectively.
    traveltimes : array-like
        Traveltimes for the station.

    """

    with open(f"{fname}.hdr", "r") as f:
        # Read the grid definition line
        line = f.readline().split()
        nx, ny, nz = int(line[0]), int(line[1]), int(line[2])
        x0, y0, z0 = float(line[3]), float(line[4]), float(line[5])
        dx, dy, dz = float(line[6]), float(line[7]), float(line[8])

        # Read the station information line
        _ = f.readline().split()

        # Read the transform definition line
        line = f.readline().split()
        cproj = Proj(proj="longlat", ellps="WGS84", datum="WGS84", no_defs=True)
        gproj = None
        if line[1] == "NONE" and not ignore_proj:
            logging.info("\tNo projection selected.")
        elif line[1] == "SIMPLE":
            orig_lat = float(line[3])
            orig_lon = float(line[5])

            # The simple projection is the Plate Carree/Equidistant Cylindrical
            gproj = Proj(proj="eqc", lat_0=orig_lat, lon_0=orig_lon, units="km")
        elif line[1] == "LAMBERT":
            proj_ellipsoid = line[3]
            orig_lat = float(line[5])
            orig_lon = float(line[7])
            parallel_1 = float(line[9])
            parallel_2 = float(line[11])

            if proj_ellipsoid == "WGS-84":
                proj_ellipsoid = "WGS84"
            elif proj_ellipsoid == "GRS-80":
                proj_ellipsoid = "GRS80"
            elif proj_ellipsoid == "WGS-72":
                proj_ellipsoid = "WGS72"
            elif proj_ellipsoid == "Australian":
                proj_ellipsoid = "aust_SA"
            elif proj_ellipsoid == "Krasovsky":
                proj_ellipsoid = "krass"
            elif proj_ellipsoid == "International":
                proj_ellipsoid = "intl"
            elif proj_ellipsoid == "Hayford-1909":
                proj_ellipsoid = "intl"
            elif proj_ellipsoid == "Clarke-1880":
                proj_ellipsoid = "clrk80"
            elif proj_ellipsoid == "Clarke-1866":
                proj_ellipsoid = "clrk66"
            elif proj_ellipsoid == "Airy":
                proj_ellipsoid = "airy"
            elif proj_ellipsoid == "Bessel":
                proj_ellipsoid = "bessel"
            elif proj_ellipsoid == "Hayford-1830":
                proj_ellipsoid = "evrst30"
            elif proj_ellipsoid == "Sphere":
                proj_ellipsoid = "sphere"
            else:
                logging.info(
                    f"Projection Ellipsoid {proj_ellipsoid} not supported!\n\tPlease "
                    "notify the QuakeMigrate developers.\n\n\tWGS-84 used instead...\n"
                )
                proj_ellipsoid = "WGS84"

            gproj = Proj(
                proj="lcc",
                lon_0=orig_lon,
                lat_0=orig_lat,
                lat_1=parallel_1,
                lat_2=parallel_2,
                units="km",
                ellps=proj_ellipsoid,
                datum="WGS84",
            )
        elif line[1] == "TRANS_MERC":
            orig_lat = float(line[5])
            orig_lon = float(line[7])

            gproj = Proj(proj="tmerc", lon=orig_lon, lat=orig_lat, units="km")

        transform = [gproj, cproj, line[1]]

    with open(f"{fname}.buf", "rb") as f:
        npts = nx * ny * nz
        buf = f.read(npts * 4)
        traveltimes = struct.unpack("f" * npts, buf)

    traveltimes = np.array(traveltimes).reshape((nx, ny, nz))
    gridspec = np.array([[nx, ny, nz], [x0, y0, z0], [dx, dy, dz]])

    return gridspec, transform, traveltimes


def _bilinear_interpolate(xz, xz_origin, xz_dimensions, traveltimes):
    """
    Perform a bi-linear interpolation between 4 data points on the input 2-D lookup
    table to calculate the traveltime to nodes on the 3-D grid.

    Note: x is used to denote the horizontal dimension over which the interpolation is
    performed. Due to the NonLinLoc definition, this corresponds to the y component of
    the grid.

    Parameters
    ----------
    xz : array-like
        Column-stacked array of distances from the station and depths for all points in
        grid.
    xz_origin : array-like
        The x (actually y) and z values of the grid origin.
    xz_dimensions : array-like
        The x (actually y) and z values of the node spacing.
    traveltimes : array-like
        A slice through the traveltime grid at x = 0, on which to perform the
        interpolation.

    Returns
    -------
    int_traveltimes : array-like
        Interpolated 3-D traveltime lookup table.

    """

    # Get indices of the nearest node
    i, k = np.floor((xz - xz_origin) / xz_dimensions).astype(int).T

    # Get fractional distance along each axis
    x_d, z_d = (np.remainder(xz, xz_dimensions) / xz_dimensions).T

    # Get 4 data points of surrounding square
    c00 = traveltimes[i, k]
    c10 = traveltimes[i + 1, k]
    c11 = traveltimes[i + 1, k + 1]
    c01 = traveltimes[i, k + 1]

    # Interpolate along x-axis
    c0 = c00 * (1 - x_d) + c10 * x_d
    c1 = c01 * (1 - x_d) + c11 * x_d

    # Interpolate along z-axis (c = c0 if np.all(z_d == 0.))
    return c0 * (1 - z_d) + c1 * z_d


def _vmodel_string(vmodel, block_model, phase):
    """
    Creates a string representation of the velocity model for the control file.

    Parameters
    ----------
    vmodel : `pandas.DataFrame` object
        DataFrame containing the velocity model to be used to generate the LUT.
        Columns:
            "Depth" of each layer in model (positive down), in km.
            "V<phase>" velocity for each layer in model (e.g. "Vp"), in km / s.
    block_model : bool
        Toggle to choose whether to interpret velocity model with constant velocity
        blocks or a linear gradient.
    phase : str
        The seismic phase for which to calculate traveltimes.

    Returns
    -------
    str_vmodel : str
        NonLinLoc formatted string describing the velocity model.

    """

    string_template = "LAYER  {0:f} {1:f} {2:f} {1:f} {2:f} 0.0 0.0"

    out = []

    i = 0
    while i < len(vmodel):
        if not block_model:
            try:
                dvdx = _velocity_gradient(i, vmodel, phase)
            except KeyError:
                dvdx = 0.0
        else:
            dvdx = 0.0

        out.append(
            string_template.format(
                vmodel["Depth"][i], vmodel[f"V{phase.lower()}"][i], dvdx
            )
        )
        i += 1

    return "\n".join(out)


def _velocity_gradient(i, vmodel, phase):
    """
    Calculate the linear gradient of the velocity model between two layers.

    Parameters
    ----------
    i : int
        Index of upper layer.
    vmodel : `pandas.DataFrame` object
        DataFrame containing the velocity model to be used to generate the LUT.
        Columns:
            "Depth" of each layer in model (positive down), in km.
            "V<phase>" velocity for each layer in model (e.g. "Vp"), in km / s.
    phase : str
        The seismic phase for which to calculate traveltimes.

    Returns
    -------
    dvdx : float
        Velocity gradients for the linear gradient model.

    """

    d_depth = vmodel["Depth"][i + 1] - vmodel["Depth"][i]
    d_vel = vmodel[f"V{phase.lower()}"][i + 1] - vmodel[f"V{phase.lower()}"][i]

    return d_vel / d_depth


def _grid_string(max_dist, depth_limits, dx):
    """
    Creates a string representation of the grid for the control file.

    Parameters
    ----------
    max_dist : float
        Maximum distance between the station and any point in the grid, in km.
    depth_limits : array-like
        Minimum/maximum extent of the grid in the z-dimension, in km.
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

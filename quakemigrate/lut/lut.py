# -*- coding: utf-8 -*-
"""
Module to produce traveltime lookup tables defined on a Cartesian grid.

:copyright:
    2020–2023, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import copy
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.interpolate import RegularGridInterpolator


class Grid3D:
    """
    A grid object represents a collection of points in a 3-D Cartesian space that can be
    used to produce regularised traveltime lookup tables that sample the continuous
    traveltime space for each station in a seismic network.

    This class also provides the series of transformations required to move between the
    input projection, the grid projection and the grid index coordinate spaces.

    The size and shape specifications of the grid are defined by providing the (input
    projection) coordinates for the lower-left and upper-right corners, a node spacing
    and the projections (defined using pyproj) of the input and grid spaces.

    Attributes
    ----------
    coord_proj : `pyproj.Proj` object
        Input coordinate space projection.
    grid_corners : array-like, shape (8, 3)
        Positions of the corners of the grid in the grid coordinate space.
    grid_proj : `pyproj.Proj` object
        Grid space projection.
    grid_xyz : array-like, shape (3, nx, ny, nz)
        Positions of the grid nodes in the grid coordinate space. The shape of each
        element of the list is defined by the number of nodes in each dimension.
    ll_corner : array-like, [float, float, float]
        Location of the lower-left corner of the grid in the grid projection. Should
        also contain the minimum depth in the grid.
    node_count : array-like, [int, int, int]
        Number of nodes in each dimension of the grid. This is calculated by finding the
        number of nodes with a given node spacing that fit between the lower-left and
        upper-right corners. This value is rounded up if the number of nodes returned is
        non-integer, to ensure the requested area is included in the grid.
    node_spacing : array-like, [float, float, float]
        Distance between nodes in each dimension of the grid.
    precision : list of float
        An appropriate number of decimal places for distances as a function of the node
        spacing and coordinate projection.
    unit_conversion_factor : float
        A conversion factor based on the grid projection, used to convert between units
        of metres and kilometres.
    unit_name : str
        Shorthand string for the units of the grid projection.
    ur_corner : array-like, [float, float, float]
        Location of the upper-right corner of the grid in the grid projection. Should
        also contain the maximum depth in the grid.

    Methods
    -------
    coord2grid(value, inverse=False, clip=False)
        Provides a transformation between the input projection and grid coordinate
        spaces.
    decimate(df, inplace=False)
        Downsamples the traveltime lookup tables by some decimation factor.
    index2coord(value, inverse=False, unravel=False, clip=False)
        Provides a transformation between grid indices (can be a flattened index or an
        [i, j, k] position) and the input projection coordinate space.
    index2grid(value, inverse=False, unravel=False)
        Provides a transformation between grid indices (can be a flattened index or an
        [i, j, k] position) and the grid coordinate space.

    """

    def __init__(self, ll_corner, ur_corner, node_spacing, grid_proj, coord_proj):
        """Instantiate the Grid3D object."""

        self.grid_proj = grid_proj
        self.coord_proj = coord_proj

        # Transform the geographical grid corners into grid coordinates
        self.ll_corner = self.coord2grid(ll_corner)[0]
        self.ur_corner = self.coord2grid(ur_corner)[0]

        # Calculate the grid dimensions and the number of nodes required
        grid_dims = self.ur_corner - self.ll_corner
        self.node_spacing = node_spacing
        self.node_count = np.ceil(grid_dims / self.node_spacing) + 1

    def decimate(self, df, inplace=False):
        """
        Resample the traveltime lookup tables by decimation by some factor.

        Parameters
        ----------
        df : array-like [int, int, int]
            Decimation factor in each dimension.
        inplace : bool, optional
            Perform the operation on the lookup table object or a copy.

        Returns
        -------
        grid : :class:`~quakemigrate.lut.lut.Grid3D` object (optional)
            Returns a Grid3D object with decimated traveltime lookup tables.

        """

        df = np.array(df, dtype=int)

        new_node_count = 1 + (self.node_count - 1) // df
        c1 = (self.node_count - df * (new_node_count - 1) - 1) // 2

        if inplace:
            grid = self
        else:
            grid = copy.deepcopy(self)

        grid.node_count = new_node_count
        grid.node_spacing = self.node_spacing * df

        for station, map_ in grid.traveltimes.items():
            for phase, ttimes in map_.items():
                grid[station][phase] = ttimes[
                    c1[0] :: df[0], c1[1] :: df[1], c1[2] :: df[2]
                ]

        if not inplace:
            return grid

    def index2grid(self, value, inverse=False, unravel=False):
        """
        Convert between grid indices and grid coordinate space.

        Parameters
        ----------
        value : array-like
            Array (of arrays) containing the grid indices (grid coordinates) to be
            transformed. Can be an array of flattened indices.
        inverse : bool, optionale
            Reverses the direction of the transform.
            Default indices -> grid coordinates.
        unravel : bool, optional
            Convert a flat index or array of flat indices into a tuple of coordinate
            arrays.

        Returns
        -------
        out : array-like
            Returns an array of arrays of the transformed values.

        """

        if unravel:
            value = np.column_stack(np.unravel_index(value, self.node_count))
        else:
            value = np.array(value)

        if inverse:
            out = np.rint((value - self.ll_corner) / self.node_spacing)
            out = np.vstack(out.astype(int))
        else:
            out = np.vstack(self.ll_corner + (value * self.node_spacing))

        # Handle cases where only a single ijk index is requested
        if out.shape[1] == 1:
            out = out.T

        return out

    def coord2grid(self, value, inverse=False):
        """
        Convert between input coordinate space and grid coordinate space.

        Parameters
        ----------
        value : array-like
            Array (of arrays) containing the coordinate locations to be transformed.
            Each sub-array should describe a single point in the 3-D input space.
        inverse : bool, optional
            Reverses the direction of the transform.
            Default input coordinates -> grid coordinates

        Returns
        -------
        out : array-like
            Returns an array of arrays of the transformed values.

        """

        v1, v2, v3 = np.array(value).T

        if inverse:
            transformer = Transformer.from_proj(self.grid_proj, self.coord_proj)
        else:
            transformer = Transformer.from_proj(self.coord_proj, self.grid_proj)

        return np.column_stack(transformer.transform(v1, v2, v3))

    def index2coord(self, value, inverse=False, unravel=False):
        """
        Convert between grid indices and input coordinate space.

        This is a utility function that wraps the other two defined transforms.

        Parameters
        ----------
        value : array-like
            Array (of arrays) containing the grid indices (grid coordinates) to be
            transformed. Can be an array of flattened indices.
        inverse : bool, optional
            Reverses the direction of the transform.
            Default indices -> input projection coordinates.
        unravel : bool, optional
            Convert a flat index or array of flat indices into a tuple of coordinate
            arrays.

        Returns
        -------
        out : array-like
            Returns an array of arrays of the transformed values.

        """

        if inverse:
            value = self.coord2grid(value)
            out = self.index2grid(value, inverse=True)
        else:
            value = self.index2grid(value, unravel=unravel)
            out = self.coord2grid(value, inverse=True)

        return out

    @property
    def node_count(self):
        """Get and set the number of nodes in each dimension of the grid."""

        try:
            return self._node_count
        except AttributeError:
            print(
                "FutureWarning: The internal data structure of LUT has changed.\nTo "
                "remove this warning you will need to convert your lookup table to the "
                "new-style\nusing `quakemigrate.lut.update_lut`."
            )
            return self._cell_count

    @node_count.setter
    def node_count(self, value):
        value = np.array(value, dtype="int32")
        assert np.all(value > 0), "Node count must be greater than [0]"
        self._node_count = value

    @property
    def node_spacing(self):
        """Get and set the spacing of nodes in each dimension of the grid."""

        try:
            return self._node_spacing
        except AttributeError:
            print(
                "FutureWarning: The internal data structure of LUT has changed.\nTo "
                "remove this warning you will need to convert your lookup table to the "
                "new-style\nusing `quakemigrate.lut.update_lut`."
            )
            return self._cell_size

    @node_spacing.setter
    def node_spacing(self, value):
        value = np.array(value, dtype="float64")
        if value.size == 1:
            value = np.repeat(value, 3)
        else:
            assert value.shape == (3,), "Node spacing must be an nx3 array."
        assert np.all(value > 0), "Node spacing must be greater than [0]"
        self._node_spacing = value

    @property
    def grid_corners(self):
        """Get the xyz positions of the nodes on the corners of the grid."""

        c = self.node_count - 1
        i, j, k = np.meshgrid([0, c[0]], [0, c[1]], [0, c[2]], indexing="ij")

        return self.index2grid(np.c_[i.flatten(), j.flatten(), k.flatten()])

    def get_grid_extent(self, cells=False):
        """
        Get the minimum/maximum extent of each dimension of the grid.

        The default returns the grid extent as the convex hull of the grid nodes. It is
        useful, for visualisation purposes, to also be able to determine the grid extent
        as the convex hull of a grid of cells centred on the grid nodes.

        Parameters
        ----------
        cells : bool, optional
            Specifies the grid mode (nodes / cells) for which to calculate the extent.

        Returns
        -------
        extent : array-like
            Pair of arrays representing two corners for the grid.

        """

        ll, ur = self.grid_corners[0], self.grid_corners[-1]

        if cells is True:
            ll -= self.node_spacing / 2
            ur += self.node_spacing / 2

        return self.coord2grid([ll, ur], inverse=True)

    grid_extent = property(get_grid_extent)

    @property
    def grid_xyz(self):
        """Get the xyz positions of all of the nodes in the grid."""

        nc = self.node_count
        ijk = np.meshgrid(*[np.arange(n) for n in nc], indexing="ij")
        xyz = self.index2grid(np.column_stack([dim.flatten() for dim in ijk]))

        return [xyz[:, dim].reshape(nc) for dim in range(3)]

    @property
    def precision(self):
        """
        Get appropriate number of decimal places as a function of the node spacing and
        coordinate projection.

        """

        return [
            -int(np.format_float_scientific(axis).split("e")[1])
            for axis in np.subtract(*self.index2coord([[0, 0, 0], [1, 1, 1]]))
        ]

    @property
    def unit_conversion_factor(self):
        """Expose unit_conversion_factor of the grid projection."""

        return self.grid_proj.crs.axis_info[0].unit_conversion_factor

    @property
    def unit_name(self):
        """Expose unit_name of the grid_projection and return shorthand."""

        unit_name = self.grid_proj.crs.axis_info[0].unit_name

        return "km" if unit_name == "kilometre" else "m"

    # --- Deprecation handling ---
    @property
    def cell_count(self):
        """Handler for deprecated attribute name 'cell_count'"""
        return self.node_count

    @cell_count.setter
    def cell_count(self, value):
        if value is None:
            return
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, change:\n"
            "\t'cell_count' -> 'node_count'"
        )
        self.node_count = value

    @property
    def cell_size(self):
        """Handler for deprecated attribute name 'cell_size'"""
        return self.node_spacing

    @cell_size.setter
    def cell_size(self, value):
        if value is None:
            return
        print(
            "FutureWarning: Parameter name has changed - continuing.\n"
            "To remove this message, change:\n"
            "\t'cell_size' -> 'node_spacing'"
        )
        self.node_spacing = value


class LUT(Grid3D):
    """
    A lookup table (LUT) object is a simple data structure that is used to store a
    series of regularised tables that, for each seismic station in a network, store the
    traveltimes to every point in the 3-D volume. These lookup tables are pre-computed
    to efficiently carry out the migration.

    This class provides utility functions that can be used to serve up or query these
    pre-computed lookup tables.

    This object is-a :class:`~quakemigrate.lut.lut.Grid3D`.

    Attributes
    ----------
    fraction_tt : float
        An estimate of the uncertainty in the velocity model as a function of a fraction
        of the traveltime. (Default 0.1 == 10%)
    max_traveltime : float
        The maximum traveltime between any station and a point in the grid.
    phases : list of str
        Seismic phases for which there are traveltime lookup tables available.
    stations_xyz : array-like, shape (n, 3)
        Positions of the stations in the grid coordinate space.
    traveltimes : dict
        A dictionary containing the traveltime lookup tables. The structure of
        this dictionary is:
            traveltimes
                - "<Station1-ID>"
                    - "<PHASE>"
                    - "<PHASE>"
                - "<Station2-ID"
                    - "<PHASE>"
                    - "<PHASE>"
                etc
    velocity_model : `pandas.DataFrame` object
        Contains the input velocity model specification.

    Methods
    -------
    serve_traveltimes(sampling_rate)
        Serve up the traveltime lookup tables.
    traveltime_to(phase, ijk)
        Query traveltimes to a grid location (in terms of indices) for a particular
        phase.
    save(filename)
        Dumps the current state of the lookup table object to a pickle file.
    load(filename)
        Restore the state of the saved LUT object from a pickle file.
    plot(fig, gs, slices=None, hypocentre=None, station_clr="k")
        Plot cross-sections of the LUT with station locations. Optionally plot slices
        through a coalescence image.

    """

    def __init__(self, fraction_tt=0.1, lut_file=None, **grid_spec):
        """Instantiate the LUT object."""

        if grid_spec:
            super().__init__(**grid_spec)
            self.fraction_tt = fraction_tt
            self.traveltimes = {}
            self.phases = []
            self.velocity_model = ""
        else:
            self.fraction_tt = fraction_tt
            self.phases = ["P", "S"]  # Handle old lookup tables
            if lut_file is not None:
                self.load(lut_file)

        self.station_data = pd.DataFrame()

    def __str__(self):
        """Return short summary string of the lookup table object."""

        ll, *_, ur = self.coord2grid(self.grid_corners, inverse=True)

        out = (
            "QuakeMigrate traveltime lookup table\nGrid parameters"
            "\n\tLower-left corner  : {lat1:10.5f}\u00b0N "
            "{lon1:10.5f}\u00b0E {dep1:10.3f} {unit_name:s}"
            "\n\tUpper-right corner : {lat2:10.5f}\u00b0N "
            "{lon2:10.5f}\u00b0E {dep2:10.3f} {unit_name:s}"
            f"\n\tNumber of nodes    : {self.node_count}"
            f"\n\tNode spacing       : {self.node_spacing} {self.unit_name}"
            "\n\n"
        )

        out = out.format(
            lat1=ll[0],
            lon1=ll[1],
            dep1=ll[2],
            lat2=ur[0],
            lon2=ur[1],
            dep2=ur[2],
            unit_name=self.unit_name,
        )

        out += "\tVelocity model:\n\t{}".format(
            str(self.velocity_model).replace("\n", "\n\t")
        )

        return out

    def serve_traveltimes(self, sampling_rate, availability=None):
        """
        Serve up the traveltime lookup tables.

        The traveltimes are multiplied by the scan sampling rate and converted to
        integers.

        Parameters
        ----------
        sampling_rate : int
            Samples per second used in the scan run.
        availability : dict, optional
            Dict of stations and phases for which to serve traveltime lookup tables:
            keys "station_phase".

        Returns
        -------
        traveltimes : `numpy.ndarray` of `numpy.int`
            Stacked traveltime lookup tables for all seismic phases, stacked along the
            station axis, with shape(nx, ny, nz, nstations)

        """

        if availability is None:
            # Serve all
            traveltimes = self._serve_traveltimes(self.phases)
        else:
            traveltimes = []
            for key, available in availability.items():
                station, phase = key.split("_")
                if available == 1:
                    try:
                        traveltimes.append(self[station][phase])
                    except KeyError:
                        traveltimes.append(self[station][f"TIME_{phase}"])
            traveltimes = np.stack(traveltimes, axis=-1)
        return np.rint(traveltimes * sampling_rate).astype(np.int32)

    def traveltime_to(self, phase, ijk, station=None):
        """
        Serve up the traveltimes to a grid location for a particular phase.

        Parameters
        ----------
        phase : str
            The seismic phase to lookup.
        ijk : array-like
            Grid indices for which to serve traveltime.
        station : str or list-like (of str), optional
            Station or stations for which to serve traveltimes. Can be str (for a single
            station) or list / `pandas.Series` object for multiple.

        Returns
        -------
        traveltimes : array-like
            Array of interpolated traveltimes to the requested grid position.

        """

        grid = tuple([np.arange(nc) for nc in self.node_count])

        if station is None:
            traveltimes = self._serve_traveltimes([phase])
        elif isinstance(station, str):
            traveltimes = self._serve_traveltimes([phase], [station])
        else:
            traveltimes = self._serve_traveltimes([phase], station)

        interpolator = RegularGridInterpolator(
            grid, traveltimes, bounds_error=False, fill_value=None
        )

        return interpolator(ijk)[0]

    def _serve_traveltimes(self, phases, stations=None):
        """
        Utility function to serve up traveltimes for a list of phases.

        Parameters
        ----------
        phases : list of str
            List of phases for which to serve traveltime lookup tables.
        stations : list-like of str, optional
            List of stations for which to serve traveltime lookup tables.

        Returns
        -------
        traveltimes : `numpy.ndarray` of float
            Array of stacked traveltimes, per the requested phases and stations.

        """

        stations = self.station_data["Name"].values if stations is None else stations

        traveltimes = []
        for phase in phases:
            for station in stations:
                try:
                    traveltimes.append(self[station][phase])
                except KeyError:
                    traveltimes.append(self[station][f"TIME_{phase}"])
        return np.stack(traveltimes, axis=-1)

    def save(self, filename):
        """
        Dump the current state of the lookup table object to a pickle file.

        Parameters
        ----------
        filename : str
            Path to location to save pickled lookup table.

        """

        # Ensure the output path exists
        pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f, 4)

    def load(self, filename):
        """
        Read the contents of a pickle file and restore state of the lookup table object.

        Parameters
        ----------
        filename : str
            Path to pickle file to load.

        """

        print(
            "FutureWarning: This method of reading lookup tables has been deprecated.\n"
            "To remove this warning:\n\tUse "
            "'quakemigrate.io.read_lut(lut_file=/path/to/file'"
        )

        with open(filename, "rb") as f:
            self.__dict__.update(pickle.load(f))

        if hasattr(self, "maps"):
            print(
                "FutureWarning: The internal data structure of LUT has changed.\nTo "
                "remove this warning you will need to convert your lookup table to the "
                "new-style\nusing `quakemigrate.lut.update_lut`."
            )

    def plot(
        self, fig, gs, slices=None, hypocentre=None, station_clr="k", station_list=None
    ):
        """
        Plot the lookup table for a particular station.

        Parameters
        ----------
        fig : `matplotlib.Figure` object
            Canvas on which LUT is plotted.
        gs : tuple(int, int)
            Grid specification for the plot.
        slices : array of arrays, optional
            Slices through a coalescence image to plot.
        hypocentre : array of floats
            Event hypocentre - will add cross-hair to plot.
        station_clr : str, optional
            Plot the stations with a particular colour.
        station_list : list-like of str, optional
            List of stations from the LUT to plot - useful if only a subset have been
            selected to be used in e.g. locate.

        """

        xy = plt.subplot2grid(gs, (2, 0), colspan=5, rowspan=5, fig=fig)
        xz = plt.subplot2grid(gs, (7, 0), colspan=5, rowspan=2, fig=fig)
        yz = plt.subplot2grid(gs, (2, 5), colspan=2, rowspan=5, fig=fig)

        xz.sharex(xy)
        yz.sharey(xy)

        # --- Set aspect ratio ---
        # Aspect is defined such that a circle will be stretched so that its
        # height is aspect times the width.
        cells_extent = self.get_grid_extent(cells=True)
        extent = abs(cells_extent[1] - cells_extent[0])
        # NOTE: no fenceposts here, because we want the size of the grid as
        # cells
        grid_size = self.node_spacing * self.node_count
        aspect = (extent[0] * grid_size[1]) / (extent[1] * grid_size[0])
        xy.set_aspect(aspect=aspect)

        bounds = np.stack(cells_extent, axis=-1)
        for i, j, ax in [(0, 1, xy), (0, 2, xz), (2, 1, yz)]:
            gminx, gmaxx = bounds[i]
            gminy, gmaxy = bounds[j]

            ax.set_xlim([gminx, gmaxx])
            ax.set_ylim([gminy, gmaxy])

            # --- Plot crosshair for event hypocentre ---
            if hypocentre is not None:
                ax.axvline(x=hypocentre[i], ls="--", lw=1.5, c="white")
                ax.axhline(y=hypocentre[j], ls="--", lw=1.5, c="white")

            # --- Plot slices through coalescence volume ---
            if slices is None:
                continue

            slice_ = slices[i + j - 1]
            nx, ny = [dim + 1 for dim in slice_.shape]
            grid1, grid2 = np.mgrid[gminx : gmaxx : nx * 1j, gminy : gmaxy : ny * 1j]
            sc = ax.pcolormesh(grid1, grid2, slice_, edgecolors="face")

            if i + j - 1 != 0:
                continue

            # --- Add colourbar ---
            cax = plt.subplot2grid(gs, (7, 5), colspan=2, rowspan=2, fig=fig)
            cax.set_axis_off()
            cb = fig.colorbar(
                sc, ax=cax, orientation="horizontal", fraction=0.8, aspect=8
            )
            cb.ax.set_xlabel("Normalised coalescence\nvalue", rotation=0, fontsize=14)

        # --- Plot stations ---
        if station_list is not None:
            station_data = self.station_data[
                self.station_data["Name"].isin(station_list)
            ]
        else:
            station_data = self.station_data
        xy.scatter(
            station_data.Longitude.values,
            station_data.Latitude.values,
            s=15,
            marker="^",
            zorder=20,
            c=station_clr,
        )
        xz.scatter(
            station_data.Longitude.values,
            station_data.Elevation.values,
            s=15,
            marker="^",
            zorder=20,
            c=station_clr,
        )
        yz.scatter(
            station_data.Elevation.values,
            station_data.Latitude.values,
            s=15,
            marker="<",
            zorder=20,
            c=station_clr,
        )
        for i, row in station_data.iterrows():
            xy.annotate(
                row["Name"],
                [row.Longitude, row.Latitude],
                zorder=20,
                c=station_clr,
                clip_on=True,
            )

        # --- Add scalebar ---
        num_cells = np.ceil(self.node_count[0] / 10)
        length = num_cells * self.node_spacing[0]
        size = extent[0] * length / grid_size[0]
        scalebar = AnchoredSizeBar(
            xy.transData,
            size=size,
            label=f"{length} {self.unit_name}",
            loc="lower right",
            pad=0.5,
            sep=5,
            frameon=False,
            color=station_clr,
        )
        xy.add_artist(scalebar)

        # --- Axes labelling ---
        xy.tick_params(
            which="both",
            left=True,
            right=True,
            top=True,
            bottom=True,
            labelleft=True,
            labeltop=True,
            labelright=False,
            labelbottom=False,
        )
        xy.set_ylabel("Latitude (deg)", fontsize=14)
        xy.yaxis.set_label_position("left")

        xz.invert_yaxis()
        xz.tick_params(
            which="both",
            left=True,
            right=True,
            top=True,
            bottom=True,
            labelleft=True,
            labeltop=False,
            labelright=False,
            labelbottom=True,
        )
        xz.set_xlabel("Longitude (deg)", fontsize=14)
        xz.set_ylabel(f"Depth ({self.unit_name})", fontsize=14)
        xz.yaxis.set_label_position("left")

        yz.tick_params(
            which="both",
            left=True,
            right=True,
            top=True,
            bottom=True,
            labelleft=False,
            labeltop=True,
            labelright=True,
            labelbottom=True,
        )
        yz.set_xlabel(f"Depth ({self.unit_name})", fontsize=14)
        yz.xaxis.set_label_position("bottom")

    @property
    def max_extent(self):
        """Get the minimum/maximum geographical extent of the stations/grid."""

        stat_min, stat_max = self.station_extent
        grid_min, grid_max = self.get_grid_extent(cells=True)

        min_extent = [min(a, b) for a, b in zip(stat_min, grid_min)]
        max_extent = [max(a, b) for a, b in zip(stat_max, grid_max)]
        diff = abs(np.subtract(max_extent, min_extent))

        min_extent = np.subtract(min_extent, 0.05 * diff)
        max_extent = np.add(max_extent, 0.05 * diff)

        return np.array([min_extent, max_extent])

    @property
    def max_traveltime(self):
        """Get the maximum traveltime from any station across the grid."""

        return np.max(self._serve_traveltimes(self.phases))

    @property
    def station_extent(self):
        """Get the minimum/maximum extent of the seismic network."""

        coordinates = self.station_data[["Longitude", "Latitude", "Elevation"]]

        return [[f(dim) for dim in coordinates.values.T] for f in (min, max)]

    @property
    def stations_xyz(self):
        """Get station locations in the grid space [X, Y, Z]."""

        coordinates = self.station_data[["Longitude", "Latitude", "Elevation"]]

        return self.coord2grid(coordinates.values)

    def __add__(self, other):
        """
        Define behaviour for the rich addition operator, "+".

        Two lookup tables which have identical grid definitions (as per "==") can be
        combined by adding the traveltime lookup tables from other.traveltimes for which
        the station key is not already in self.traveltimes.

        Parameters
        ----------
        other : :class:`~quakemigrate.lut.lut.LUT` object
            LUT with traveltime lookup tables to add to self.

        """

        if not isinstance(other, LUT):
            print("Addition not defined for non-LUT object.")
            return self
        else:
            if self == other:
                self.traveltimes.update(other.traveltimes)
                return self
            else:
                print("Grid definitions do not match - cannot combine.")

    def __eq__(self, other):
        """
        Define behaviour for the rich equality operator, "==".

        Two lookup tables are defined to be equal if their grid definitions are
        identical - corners, node spacing, projections.

        Parameters
        ----------
        other : :class:`~quakemigrate.lut.lut.LUT` object
            LUT with which to test equality with self.

        """

        # Test if other isinstance of LUT
        if not isinstance(other, LUT):
            print("Equality of LUT with non-LUT object is undefined.")
            return False
        else:
            # Test equality of grid corners
            eq_corners = (self.grid_corners == other.grid_corners).all()

            # Test equality of node spacings
            eq_sizes = (self.node_spacing == other.node_spacing).all()

            # Test equality of projections
            eq_projections = (
                self.grid_proj == other.grid_proj
                and self.coord_proj == other.coord_proj
            )

            return eq_corners and eq_sizes and eq_projections

    def __getitem__(self, key):
        """
        Provide a method to directly access traveltime tables by station key without
        having to go through the traveltimes dictionary.

        Parameters
        ----------
        key : str
            Station ID for which to search.

        Returns
        -------
        station_traveltimes : dict
            Traveltime lookup table for key (station), if key exists.

        """

        try:
            return self.traveltimes[key]
        except AttributeError:
            return self.maps[key]
        except KeyError:
            print(f"No traveltime lookup table available for '{key}'.")

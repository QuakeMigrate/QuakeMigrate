# -*- coding: utf-8 -*-
"""
Module to produce traveltime lookup tables defined on a Cartesian grid.

"""

import copy
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from scipy.interpolate import RegularGridInterpolator


class Grid3D:
    """
    A grid object represents a collection of points in a 3-D Cartesian space
    that can be used to produce regularised traveltime lookup tables that
    sample the continuous traveltime space for each station in a seismic
    network.

    This class also provides the series of transformations required to move
    between the input projection, the grid projection and the grid index
    coordinate spaces.

    The size and shape specifications of the grid are defined by providing the
    (input projection) coordinates for the lower-left and upper-right corners,
    a cell size and the projections (defined using pyproj) of the input and
    grid spaces.

    Attributes
    ----------
    ll_corner : array-like, [float, float, float]
        Location of the lower-left corner of the grid in the grid
        projection. Should also contain the minimum depth in the grid.
    ur_corner : array-like, [float, float, float]
        Location of the upper-right corner of the grid in the grid
        projection. Should also contain the maximum depth in the grid.
    cell_size : array-like, [float, float, float]
        Size of a cell in each dimension of the grid.
    grid_proj : pyproj Proj object
        Grid space projection.
    coord_proj : pyproj Proj object
        Input coordinate space projection.
    cell_count : array-like, [int, int, int]
        Number of cells in each dimension of the grid. This is calculated by
        finding the number of cells with cell_size will fit between the
        lower-left and upper-right corners. This value is rounded up if the
        number of cells returned is non-integer, to ensure the requested area
        is included in the grid.
    grid_corners : array-like, shape (8, 3)
        Positions of the corners of the grid in the grid coordinate space.
    grid_xyz : array-like, shape (3,)
        Positions of the grid nodes in the grid coordinate space. The shape of
        each element of the list is defined by the number of cells in each
        dimension.
    stations_xyz : array-like, shape (n, 3)
        Positions of the stations in the grid coordinate space.

    Methods
    -------
    decimate(df, inplace=False)
        Downsamples the traveltime lookup tables by some decimation factor.
    index2grid(value, inverse=False, unravel=False)
        Provides a transformation between grid indices (can be a flattened
        index or an [i, j, k] position) and the grid coordinate space.
    coord2grid(value, inverse=False, clip=False)
        Provides a transformation between the input projection and grid
        coordinate spaces.
    index2coord(value, inverse=False, unravel=False, clip=False)
        Provides a transformation between grid dindices (can be a flattened
        index or an [i, j, k] position) and the input projection coordinate
        space.

    """

    def __init__(self, ll_corner, ur_corner, cell_size, grid_proj, coord_proj):
        """Instantiate the Grid3D object."""

        self.grid_proj = grid_proj
        self.coord_proj = coord_proj

        # Transform the geographical grid corners into grid coordinates
        self.ll_corner = self.coord2grid(ll_corner)[0]
        self.ur_corner = self.coord2grid(ur_corner)[0]

        # Calculate the grid dimensions and the number of cells required
        grid_dims = self.ur_corner - self.ll_corner
        self.cell_size = cell_size
        self.cell_count = np.ceil(grid_dims / self.cell_size) + 1

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
        grid : Grid3D object (optional)
            Returns a Grid3D object with decimated traveltime lookup tables.

        """

        df = np.array(df, dtype=np.int)

        new_cell_count = 1 + (self.cell_count - 1) // df
        c1 = (self.cell_count - df * (new_cell_count - 1) - 1) // 2

        if inplace:
            grid = self
        else:
            grid = copy.deepcopy(self)

        grid.cell_count = new_cell_count
        grid.cell_size = self.cell_size * df

        for station, map_ in grid.maps.items():
            for phase, ttimes in map_.items():
                grid[station][phase] = ttimes[c1[0]::df[0],
                                              c1[1]::df[1],
                                              c1[2]::df[2]]

        if not inplace:
            return grid

    def index2grid(self, value, inverse=False, unravel=False):
        """
        Convert between grid indices and grid coordinate space.

        Parameters
        ----------
        value : array-like
            Array (of arrays) containing the grid indices (grid coordinates)
            to be transformed. Can be an array of flattened indices.
        inverse : bool, optionale
            Reverses the direction of the transform.
            Default indices -> grid coordinates.
        unravel : bool, optional
            Convert a flat index or array of flat indices into a tuple of
            coordinate arrays.

        Returns
        -------
        out : array-like
            Returns an array of arrays of the transformed values.

        """

        if unravel:
            value = np.column_stack(np.unravel_index(value, self.cell_count))
        else:
            value = np.array(value)

        if inverse:
            out = np.rint((value - self.ll_corner) / self.cell_size)
            out = np.vstack(out.astype(int))
        else:
            out = np.vstack(self.ll_corner + (value * self.cell_size))

        # Handle cases where only a single ijk index is requested
        if out.shape[1] == 1:
            out = out.T

        return out

    def coord2grid(self, value, inverse=False, clip=False):
        """
        Convert between input coordinate space and grid coordinate space.

        Parameters
        ----------
        value : array-like
            Array (of arrays) containing the coordinate locations to be
            transformed. Each sub-array should describe a single point in the
            3-D input space.
        inverse : bool, optional
            Reverses the direction of the transform.
            Default input coordinates -> grid coordinates
        clip : bool, optional

        Returns
        -------
        out : array-like
            Returns an array of arrays of the transformed values.

        """

        v1, v2, v3 = np.array(value).T

        if inverse:
            inproj, outproj = self.grid_proj, self.coord_proj
        else:
            inproj, outproj = self.coord_proj, self.grid_proj

        return np.column_stack(pyproj.transform(inproj, outproj, v1, v2, v3))

    def index2coord(self, value, inverse=False, unravel=False, clip=False):
        """
        Convert between grid indices and input coordinate space.

        This is a utility function that wraps the other two defined transforms.

        Parameters
        ----------
        value : array-like
            Array (of arrays) containing the grid indices (grid coordinates)
            to be transformed. Can be an array of flattened indices.
        inverse : bool, optional
            Reverses the direction of the transform.
            Default indices -> input projection coordinates.
        unravel : bool, optional
            Convert a flat index or array of flat indices into a tuple of
            coordinate arrays.
        clip : bool, optional

        Returns
        -------
        out : array-like
            Returns an array of arrays of the transformed values.

        """

        if inverse:
            value = self.coord2grid(value, clip=clip)
            out = self.index2grid(value, inverse=True)
        else:
            value = self.index2grid(value, unravel=unravel)
            out = self.coord2grid(value, inverse=True, clip=clip)

        return out

    @property
    def grid_corners(self):
        """Get the xyz positions of the cells on the edge of the grid."""

        c = self.cell_count - 1
        i, j, k = np.meshgrid([0, c[0]], [0, c[1]], [0, c[2]], indexing="ij")

        return self.index2grid(np.c_[i.flatten(), j.flatten(), k.flatten()])

    @property
    def grid_xyz(self):
        """Get the xyz positions of all of the cells in the grid."""

        cc = self.cell_count
        i, j, k = np.meshgrid(np.arange(cc[0]), np.arange(cc[1]),
                              np.arange(cc[2]), indexing="ij")
        xyz = self.index2grid(np.c_[i.flatten(), j.flatten(), k.flatten()])
        x, y, z = [xyz[:, dim].reshape(cc) for dim in range(3)]

        return x, y, z

    @property
    def stations_xyz(self):
        """Get station locations in the grid space [X, Y, Z]."""

        return self.coord2grid(self.station_data[["Longitude",
                                                  "Latitude",
                                                  "Elevation"]].values)

    @property
    def cell_count(self):
        """Get and set the number of cells in each dimension of the grid."""

        return self._cell_count

    @cell_count.setter
    def cell_count(self, value):
        value = np.array(value, dtype="int32")
        assert (np.all(value > 0)), "Cell count must be greater than [0]"
        self._cell_count = value

    @property
    def cell_size(self):
        """Get and set the size of a cell in each dimension of the grid."""

        return self._cell_size

    @cell_size.setter
    def cell_size(self, value):
        value = np.array(value, dtype="float64")
        if value.size == 1:
            value = np.repeat(value, 3)
        else:
            assert (value.shape == (3,)), "Cell size must be an n by 3 array."
        assert (np.all(value > 0)), "Cell size must be greater than [0]"
        self._cell_size = value

    @property
    def precision(self):
        """
        Get appropriate number of decimal places as a function of the
        cell size and coordinate projection.

        """

        return [-int(np.format_float_scientific(axis).split("e")[1]) for axis
                in np.subtract(*self.index2coord([[0, 0, 0], [1, 1, 1]]))]

    @property
    def unit_conversion_factor(self):
        """Expose unit_conversion_factor of the grid projection."""

        return self.grid_proj.crs.axis_info[0].unit_conversion_factor

    @property
    def unit_name(self):
        """Expose unit_name of the grid_projection and return shorthand."""

        unit_name = self.grid_proj.crs.axis_info[0].unit_name

        return "km" if unit_name == "kilometre" else "m"


class LUT(Grid3D):
    """
    A lookup table (LUT) object is a simple data structure that is used to
    store a series of regularised tables that, for each seismic station in a
    network, store the traveltimes to every point in the 3-D volume. These
    lookup tables are pre-computed to reduce the computational cost of the
    back-projection method.

    This class provides utility functions that can be used to serve up or query
    these pre-computed lookup tables.

    This object is-a Grid3D.

    Attributes
    ----------
    fraction_tt : float
        An estimate of the uncertainty in the velocity model as a function of
        a fraction of the traveltime. (Default 0.1 == 10%)
    maps : dict
        A dictionary containing the traveltime lookup tables. The structure of
        this dictionary is:
            maps
                - "<Station1-ID>"
                    - "<PHASE>"
                    - "<PHASE>"
                - "<Station2-ID"
                    - "<PHASE>"
                    - "<PHASE>"
                etc
    max_traveltime : float
        The maximum traveltime between any station and a point in the grid.
    phases : list of str
        Seismic phases for which there are traveltime lookup tables available.
    velocity_model : `pandas.DataFrame` object
        Contains the input velocity model specification.

    Methods
    -------
    traveltimes(sampling_rate)
        Serve up the traveltime lookup tables.
    traveltime_to(phase, ijk)
        Query traveltimes to a grid location (in terms of indices) for a
        particular phase.
    save(filename)
        Dumps the current state of the lookup table object to a pickle file.
    load(filename)
        Restore the state of the saved LUT object from a pickle file.
    plot(fig, gs, slices=None, hypocentre=None, station_clr="k")
        Plot cross-sections of the LUT with station locations. Optionally plot
        slices through a coalescence volume.

    """

    def __init__(self, fraction_tt=0.1, lut_file=None, **grid_spec):
        """Instantiate the LUT object."""

        if grid_spec:
            super().__init__(**grid_spec)
            self.fraction_tt = fraction_tt
            self.maps = {}
            self.phases = []
            self.velocity_model = ""
        else:
            self.fraction_tt = fraction_tt
            self.phases = ["P", "S"]  # Handle old lookup tables
            if lut_file is not None:
                self.load(lut_file)

    def __str__(self):
        """Return short summary string of the lookup table object."""

        ll, *_, ur = self.coord2grid(self.grid_corners, inverse=True)
        cc = self.cell_count
        cs = self.cell_size

        out = ("QuakeMigrate traveltime lookup table\nGrid parameters"
               "\n\tLower-left corner  : {lat1:10.5f}\u00b0N "
               "{lon1:10.5f}\u00b0E {dep1:10.3f} m"
               "\n\tUpper-right corner : {lat2:10.5f}\u00b0N "
               "{lon2:10.5f}\u00b0E {dep2:10.3f} m"
               f"\n\tNumber of cells    : {cc}"
               f"\n\tCell dimensions    : {cs} m\n\n")

        out = out.format(lat1=ll[0], lon1=ll[1], dep1=ll[2],
                         lat2=ur[0], lon2=ur[1], dep2=ur[2])

        out += ("\tVelocity model:\n"
                "\t{}".format(str(self.velocity_model).replace("\n", "\n\t")))

        return out

    def traveltimes(self, sampling_rate):
        """
        Serve up the traveltime lookup tables.

        The traveltimes are multiplied by the scan sampling rate and converted
        to integers.

        Parameters
        ----------
        sampling_rate : int
            Samples per second used in the scan run.

        Returns
        -------
        traveltimes : `numpy.ndarray` of `numpy.int`
            Stacked traveltime lookup tables for all seismic phases, stacked
            along the station axis, with shape(nx, ny, nz, nstations)

        """

        traveltimes = self._serve_traveltimes(self.phases)

        return np.rint(traveltimes * sampling_rate).astype(np.int32)

    def traveltime_to(self, phase, ijk):
        """
        Serve up the traveltimes to a grid location for a particular phase.

        Parameters
        ----------
        phase : str
            The seismic phase to lookup.
        ijk : array-like
            Grid indices for which to serve traveltime.

        Returns
        -------
        traveltimes : array-like
            Array of interpolated traveltimes to the requested grid position.

        """

        grid = tuple([np.arange(cc) for cc in self.cell_count])

        traveltimes = self._serve_traveltimes([phase])

        interpolator = RegularGridInterpolator(grid, traveltimes,
                                               bounds_error=False,
                                               fill_value=None)

        return interpolator(ijk)[0]

    @property
    def max_traveltime(self):
        """Get the maximum traveltime from any station across the grid."""

        return np.max(self._serve_traveltimes(self.phases))

    def _serve_traveltimes(self, phases):
        """Utility function to serve up traveltimes for a list of phases."""

        traveltimes = []
        for phase in phases:
            for station in self.station_data["Name"].values:
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
        Read the contents of a pickle file and restore state of the lookup
        table object.

        Parameters
        ----------
        filename : str
            Path to pickle file to load.

        """

        print("FutureWarning: This method of reading lookup tables has been"
              "deprecated.\nTo remove this warning:\n"
              "\tUse 'QMigrate.io.read_lut(lut_file=/path/to/file'")

        with open(filename, "rb") as f:
            self.__dict__.update(pickle.load(f))

    def plot(self, fig, gs, slices=None, hypocentre=None, station_clr="k"):
        """
        Plot the lookup table for a particular station.

        Parameters
        ----------
        fig : `matplotlib.Figure` object
            Canvas on which LUT is plotted.
        gs : tuple(int, int)
            Grid specification for the plot.
        slices : array of arrays, optional
            Slices through a coalescence volume to plot.
        hypocentre : array of floats
            Event hypocentre - will add cross-hair to plot.
        station_clr : str, optional
            Plot the stations with a particular colour.

        """

        xy = plt.subplot2grid(gs, (2, 0), colspan=5, rowspan=5, fig=fig)
        xz = plt.subplot2grid(gs, (7, 0), colspan=5, rowspan=2, fig=fig)
        yz = plt.subplot2grid(gs, (2, 5), colspan=2, rowspan=5, fig=fig)

        xz.get_shared_x_axes().join(xy, xz)
        yz.get_shared_y_axes().join(xy, yz)

        # --- Set bounds ---
        corners = self.coord2grid(self.grid_corners, inverse=True)
        mins = [np.min(dim) for dim in corners.T]
        maxs = [np.max(dim) for dim in corners.T]
        sizes = (np.array(maxs) - np.array(mins)) / self.cell_count
        stack = np.c_[mins, maxs, sizes]

        for idx1, idx2, ax in [(0, 1, xy), (0, 2, xz), (2, 1, yz)]:
            min1, max1, size1 = stack[idx1]
            min2, max2, size2 = stack[idx2]

            ax.set_xlim([min1, max1])
            ax.set_ylim([min2, max2])

            # --- Plot slices through coalescence volume ---
            if slices is not None:
                idx = (idx1 + idx2) - 1
                slice_ = slices[idx]
                grid1, grid2 = np.mgrid[min1:max1 + size1:size1,
                                        min2:max2 + size2:size2]
                grid1 = grid1[:slice_.shape[0]+1, :slice_.shape[1]+1]
                grid2 = grid2[:slice_.shape[0]+1, :slice_.shape[1]+1]
                sc = ax.pcolormesh(grid1, grid2, slice_, cmap="viridis",
                                   edgecolors="face")

                if idx == 0:
                    # --- Add colourbar ---
                    cax = plt.subplot2grid(gs, (2, 7), colspan=1, rowspan=5,
                                           fig=fig)
                    cax.set_axis_off()
                    cbar = fig.colorbar(sc, ax=cax, orientation="vertical",
                                        fraction=0.4)
                    cbar.ax.set_ylabel("Coalescence value", rotation=90,
                                       fontsize=14)

            # --- Plot crosshair for event hypocentre ---
            if hypocentre is not None:
                ax.axvline(x=hypocentre[idx1], ls="--", lw=1.5, c="white")
                ax.axhline(y=hypocentre[idx2], ls="--", lw=1.5, c="white")

        # --- Plot stations ---
        xy.scatter(self.station_data.Longitude, self.station_data.Latitude,
                   s=15, marker="v", zorder=20, c=station_clr)
        xz.scatter(self.station_data.Longitude, self.station_data.Elevation,
                   s=15, marker="v", zorder=20, c=station_clr)
        yz.scatter(self.station_data.Elevation, self.station_data.Latitude,
                   s=15, marker=">", zorder=20, c=station_clr)
        for i, row in self.station_data.iterrows():
            xy.annotate(row["Name"], [row.Longitude, row.Latitude], zorder=20,
                        c=station_clr)

        # --- Axes labelling ---
        xy.xaxis.tick_top()

        xz.yaxis.tick_right()
        xz.invert_yaxis()
        xz.set_xlabel("Longitude (deg)", fontsize=14)
        xz.set_ylabel(f"Depth ({self.unit_name})", fontsize=14)
        xz.yaxis.set_label_position("right")

        yz.yaxis.tick_right()
        yz.set_xlabel(f"Depth ({self.unit_name})", fontsize=14)
        yz.set_ylabel("Latitude (deg)", fontsize=14)
        yz.yaxis.set_label_position("right")

    def __add__(self, other):
        """
        Define behaviour for the rich addition operator, "+".

        Two lookup tables which have identical grid definitions (as per "==")
        can be combined by adding the traveltime lookup tables from other.maps
        for which the station key is not already in self.maps.

        Parameters
        ----------
        other : `QMigrate.lut.LUT` object
            LUT with traveltime lookup tables to add to self.

        """

        if not isinstance(other, LUT):
            print("Addition not defined for non-LUT object.")
            return self
        else:
            if self == other:
                for key, ttime in other.maps.items():
                    if key not in self.maps.keys():
                        self.maps[key] = ttime
                return self
            else:
                print("Grid definitions do not match - cannot combine.")

    def __eq__(self, other):
        """
        Define behaviour for the rich equality operator, "==".

        Two lookup tables are defined to be equal if their grid definitions are
        identical - corners, cell size, projections.

        Parameters
        ----------
        other : QuakeMigrate LUT object
            LUT with which to test equality with self.

        """

        # Test if other isinstance of LUT
        if not isinstance(other, LUT):
            print("Equality of LUT with non-LUT object is undefined.")
            return False
        else:
            # Test equality of grid corners
            eq_corners = (self.grid_corners == other.grid_corners).all()

            # Test equality of cell sizes
            eq_sizes = (self.cell_size == other.cell_size).all()

            # Test equality of projections
            eq_projections = (self.grid_proj == other.grid_proj
                              and self.coord_proj == other.coord_proj)

            return eq_corners and eq_sizes and eq_projections

    def __getitem__(self, key):
        """
        Provide a method to directly access traveltime maps by station key
        without having to go through the maps dictionary.

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
            return self.maps[key]
        except KeyError:
            print(f"No traveltime lookup table available for '{key}'.")

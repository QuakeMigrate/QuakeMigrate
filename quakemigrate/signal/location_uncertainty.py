"""
Collection of methods for computing locations and associated uncertainties from 3-D
maps of coalescence values.

:copyright:
    2020–2026, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import Rbf
from scipy.signal import fftconvolve

from quakemigrate import util


if TYPE_CHECKING:
    from collections.abc import Callable

    from quakemigrate.lut import LUT


REV_SLICE = slice(None, None, -1)


@util.timeit()
def spline_location(
    lut: LUT, coalescence_map: np.ndarray, window: int = 5, upscale: int = 10
) -> np.ndarray[float]:
    """
    Fit a 2-/3-D spline function to a region around the maximum coalescence in the
    marginalised coalescence map and interpolate by factor `upscale` to return a
    sub-grid maximum coalescence location.

    Parameters
    ----------
    lut:
        Traveltime lookup table, which includes the specification of the underlying
        grid on which migration and stacking is performed.
    coalescence_map:
        Marginalised 3-D coalescence map.
    window:
        Window of grid nodes (+/-(window-1)//2 in x, y and z) around max value in
        coalescence_map to perform the fit over.
    upscale:
        Upscaling factor to interpolate the fitted 2-/3-D spline function by.

    Returns
    -------
    location:
        Max coalescence location from spline interpolation.

    """

    grid_shape = np.array(coalescence_map.shape)
    ijk = np.array(
        np.unravel_index(np.nanargmax(coalescence_map), coalescence_map.shape)
    )
    gridded_location = lut.index2coord(ijk)[0]

    depth_constrained = grid_shape[2] == 1
    active_dims = [0, 1] if depth_constrained else [0, 1, 2]

    # Trim coalescence map to small window around maximum and upscale
    half_window = (window - 1) // 2
    lower = np.clip(ijk - half_window, 0, grid_shape)
    upper = np.clip(ijk + half_window + 1, 0, grid_shape)

    window_widths = upper[active_dims] - lower[active_dims]
    if not np.all(window_widths == window_widths[0]):
        logging.info("\t !! spline interpolation window crosses edge of grid !!")
        logging.info("\t\t\tGridded Location returned")
        return gridded_location

    slices = [slice(lower[axis], upper[axis]) for axis in range(3)]
    trimmed = coalescence_map[tuple(slices)]
    if len(active_dims) == 2:
        trimmed = trimmed[:, :, ijk[2]]

    # Define the interpolation function over the grid at input scale
    grids_in = np.meshgrid(*[np.arange(n) for n in trimmed.shape], indexing="ij")
    interpolator = Rbf(
        *[grid.ravel() for grid in grids_in], trimmed.ravel(), function="cubic"
    )

    # Interpolate the coalescence map on the upsampled grid
    grids_out = np.meshgrid(
        *[np.linspace(0, n - 1, (n - 1) * upscale + 1) for n in trimmed.shape],
        indexing="ij",
    )
    interpolated = interpolator(*[grid.ravel() for grid in grids_out]).reshape(
        grids_out[0].shape
    )

    # Get max coalescence index on interpolated grid and map to original grid space
    _spline_ijk = np.unravel_index(np.nanargmax(interpolated), interpolated.shape)
    spline_ijk = ijk.astype(float)
    for dim in active_dims:
        spline_ijk[dim] = _spline_ijk[dim] / upscale + lower[dim]

    logging.debug(f"\t\tGridded loc: {tuple(ijk)}")
    logging.debug(f"\t\tSpline  loc: {tuple(spline_ijk)}")

    spline_offset = np.abs(ijk[active_dims] - spline_ijk[active_dims])
    if np.any(spline_offset > half_window):
        logging.info("\t !! spline location outside interpolation window !!")
        logging.info("\t\t\tGridded Location returned")
        return gridded_location

    if np.any(spline_offset > 1):
        logging.debug(
            "\tspline location outside grid cell with maximum coalescence value"
        )

    return lut.index2coord(spline_ijk)[0]


@util.timeit()
def gaussian_fit(
    lut: LUT, coalescence_map: np.ndarray, threshold: float = 0.0, window: int = 7
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit a Gaussian function to a region around the maximum coalescence location in
    the 2- or 3-D marginalised coalescence map: return expectation location and
    associated uncertainty.

    Parameters
    ----------
    lut:
        Traveltime lookup table, which includes the specification of the underlying
        grid on which migration and stacking is performed.
    coalescence_map:
        Marginalised 3-D coalescence map.
    threshold:
        Cut-off value used to trim coalescence_map: only data above this value are
        retained.
    window:
        Window of grid nodes (+/-(window-1)//2 in x, y and z) around max value in
        coalescence_map to perform the fit over.

    Returns
    -------
    location:
        Best-fitting Gaussian location in LUT coordinates.
    uncertainty:
        One sigma axis-aligned uncertainties on expectation location from Gaussian
        fit.
    gaussian:
        Full rotated Gaussian parameters.

    """

    # Get shape of 3-D coalescence map and max coalescence grid location
    grid_shape = np.array(coalescence_map.shape)
    ijk = np.array(
        np.unravel_index(np.nanargmax(coalescence_map), coalescence_map.shape)
    )
    gridded_location = lut.index2coord(ijk)[0]

    # Work in active dimensions only:
    #   3-D: x = (i, j, k)
    #   2-D fixed-depth: x = (i, j)
    depth_constrained = grid_shape[2] == 1
    fit_dims = [0, 1] if depth_constrained else [0, 1, 2]

    map_to_fit = coalescence_map[:, :, ijk[2]] if depth_constrained else coalescence_map
    fit_shape = grid_shape[fit_dims]
    fit_ijk = ijk[fit_dims]

    # Select points near the coalescence peak where the Gaussian approximation
    # should be locally valid
    mask = _mask_nd(fit_shape, fit_ijk, window)
    flag = np.logical_and(map_to_fit > threshold, mask)

    indices = np.where(flag)
    if len(indices[0]) == 0:
        uncertainty = np.full(3, np.nan)
        return gridded_location, uncertainty, {}

    # Subtract mean of the coalescence map being fitted from the local grid window
    # so it is better approximated by a Gaussian (which goes to zero at infinity)
    map_to_fit = map_to_fit - np.nanmean(map_to_fit)

    # Coordinates relative to the grid maximum:
    #
    #   x' = x - x0
    #
    # The fitted centre_offset below is therefore also relative to this maximum
    centred_coords = np.vstack([idx - centre for idx, centre in zip(indices, fit_ijk)])

    # Build the linear least-squares system for:
    #
    #   y(x') = x'.T M x' + h.T x' + c
    #
    # In 2-D this fits:
    #   ax**2 + by**2 + dxy + gx + hy + c
    #
    # In 3-D this fits:
    #   ax**2 + by**2 + cz**2 + dxy + exz + fyz + gx + hy + iz + c
    design_matrix, terms = build_gaussian_design_matrix(centred_coords)

    # Fit the Gaussian exponent rather than the Gaussian itself
    #
    # If A(x') is the coalescence amplitude, then locally:
    #
    #   A(x') ~= A0 exp(-y(x'))
    #
    # so:
    #
    #   -log A(x') ~= y(x')
    #
    # where y(x') is a quadratic form
    y = -np.log(np.clip(map_to_fit.astype(np.float64)[indices], 1e-300, np.inf))

    # Solve least squares for the polynomial coefficients
    parameters = y @ np.linalg.pinv(design_matrix)

    # Convert polynomial coefficients into:
    #
    #   y(x') = x'.T M x' + h.T x' + c
    #
    # where:
    #   M = quadratic_matrix
    #   h = linear_vector
    #   c = constant
    quadratic_matrix, linear_vector, constant = unpack_gaussian_quadratic_terms(
        parameters, terms, len(fit_dims)
    )

    # The Gaussian exponent can also be written:
    #
    #   y(x') = 0.5 (x' - mu).T Sigma^-1 (x' - mu) + constant
    #
    # Therefore:
    #
    #   M = 0.5 Sigma^-1
    #   Sigma = 0.5 M^-1
    #
    # The stationary point is found from:
    #
    #   grad y(x') = 2 M x' + h = 0
    #   x' = -0.5 M^-1 h
    #
    # Equivalently:
    #
    #   precision_matrix = 2 M
    #   centre_offset = -precision_matrix^-1 h
    precision_matrix_ = 2.0 * quadratic_matrix
    centre_offset = np.linalg.solve(-precision_matrix_, linear_vector)
    precision_matrix = embed_matrix(precision_matrix_, fit_dims)

    # Evaluate the fitted quadratic at the centre to recover the Gaussian amplitude:
    #
    #   A0 = exp(-y(mu))
    k = constant - centre_offset @ quadratic_matrix @ centre_offset
    amplitude = np.exp(-k)

    # Covariance in grid-index units:
    #
    #   Sigma_grid = 0.5 M^-1
    covariance_grid = 0.5 * np.linalg.inv(quadratic_matrix)
    covariance_matrix = embed_matrix(covariance_grid, fit_dims)

    # Convert covariance from grid-index units to physical LUT units.
    #
    # If S = diag(dx, dy, dz), then:
    #
    #   Sigma_physical = S Sigma_grid S.T
    fit_spacing = lut.node_spacing[fit_dims]
    spacing_matrix = np.diag(fit_spacing)
    covariance_physical = spacing_matrix @ covariance_grid @ spacing_matrix
    covariance_matrix_physical = embed_matrix(covariance_physical, fit_dims)

    # Eigendecomposition of the physical covariance gives the uncertainty ellipsoid:
    #
    #   eigenvectors -> principal axes
    #   sqrt(eigenvalues) -> principal standard deviations
    eigvals, eigvecs = np.linalg.eigh(covariance_physical)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    principal_uncertainty = np.full(3, np.nan)
    principal_uncertainty[: len(fit_dims)] = np.sqrt(np.clip(eigvals, 1e-10, np.inf))

    principal_axes = np.full((3, 3), np.nan)
    principal_axes[np.ix_(fit_dims, range(len(fit_dims)))] = eigvecs
    if depth_constrained:
        principal_axes[2, 2] = 1.0

    # Axis-aligned standard deviations are the square roots of the covariance diagonal.
    uncertainty = np.full(3, np.nan)
    uncertainty[fit_dims] = np.sqrt(
        np.clip(np.diag(covariance_physical), 1e-10, np.inf)
    )

    # Get final Gaussian-fit location
    grid_location = ijk.astype(float)
    grid_location[fit_dims] = centre_offset + fit_ijk
    location = lut.index2coord(grid_location)[0]

    gaussian = {
        "depth_constrained": depth_constrained,
        "fit_dims": fit_dims,
        "grid_location": grid_location,
        "amplitude": amplitude,
        "quadratic_matrix": quadratic_matrix,
        "precision_matrix": precision_matrix,
        "covariance_matrix": covariance_matrix,
        "covariance_matrix_physical": covariance_matrix_physical,
        "principal_axes": principal_axes,
        "principal_uncertainty": principal_uncertainty,
        "axis_aligned_uncertainty": uncertainty,
        "eigenvalues": eigvals,
    }

    return location, uncertainty, gaussian


@util.timeit()
def covariance_fit(
    lut: LUT,
    coalescence_map: np.ndarray,
    threshold: float = 0.90,
    window: int | None = None,
) -> tuple[list[float], list[float]]:
    """
    Calculate the covariance of the marginalised coalescence map, filtered above a
    percentile threshold. Supports full 3-D and fixed-depth 2-D grids. Optionally can
    also perform the fit on a sub-window of the grid around the maximum coalescence
    location.

    Parameters
    ----------
    lut:
        Traveltime lookup table, which includes the specification of the underlying
        grid on which migration and stacking is performed.
    coalescence_map:
        Marginalised 3-D coalescence map.
    threshold:
        Cut-off threshold (fractional percentile) to trim coalescence_map; only data
        above this percentile will be retained.
    window:
        Window of grid nodes (+/-(window-1)//2 in x, y and z) around max value in
        coalescence_map to perform the fit over.

    Returns
    -------
    location:
        Expectation location from covariance fit.
    uncertainty:
        One sigma uncertainties on expectation location from covariance fit.

    """

    # Get shape of 3-D coalescence map and max coalesence grid location
    grid_shape = np.array(coalescence_map.shape)
    ijk = np.array(
        np.unravel_index(np.nanargmax(coalescence_map), coalescence_map.shape)
    )
    gridded_location = lut.index2coord(ijk)[0]

    # Work in active dimensions only:
    #   3-D: x = (i, j, k)
    #   2-D fixed-depth: x = (i, j)
    depth_constrained = grid_shape[2] == 1
    fit_dims = [0, 1] if depth_constrained else [0, 1, 2]

    map_to_fit = coalescence_map[:, :, ijk[2]] if depth_constrained else coalescence_map
    fit_shape = grid_shape[fit_dims]
    fit_ijk = ijk[fit_dims]

    # If window is specified, clip the grid to only look here.
    if window is not None:
        mask = _mask_nd(fit_shape, fit_ijk, window)
        flag = np.logical_and(map_to_fit > threshold, mask)
    else:
        flag = map_to_fit > threshold

    # Treat the coalescence values in the grid as the sample weights
    weights = map_to_fit.astype(float).ravel()
    weights[~flag.flatten()] = np.nan

    weight_sum = np.nansum(weights)
    if weight_sum == 0 or not np.isfinite(weight_sum):
        uncertainty = np.full(3, np.nan)
        return gridded_location, uncertainty

    # Get the x, y and z samples on which to perform the fit
    grids = np.meshgrid(*[np.arange(n) for n in fit_shape], indexing="ij")

    fit_spacing = lut.node_spacing[fit_dims]
    samples = [grid.ravel() * spacing for grid, spacing in zip(grids, fit_spacing)]

    expectation = np.array(
        [np.nansum(weights * sample) / weight_sum for sample in samples]
    )

    covariance_fit = np.zeros((len(fit_dims), len(fit_dims)))
    for i in range(len(fit_dims)):
        for j in range(i, len(fit_dims)):
            covariance_fit[i, j] = (
                np.nansum(
                    weights
                    * (samples[i] - expectation[i])
                    * (samples[j] - expectation[j])
                )
                / weight_sum
            )
            covariance_fit[j, i] = covariance_fit[i, j]

    location_xyz = lut.ll_corner.copy()
    location_xyz[fit_dims] += expectation

    if depth_constrained:
        location_xyz[2] = lut.ll_corner[2]

    location = lut.coord2grid(location_xyz, inverse=True)[0]

    covariance_matrix = embed_matrix(covariance_fit, fit_dims)

    uncertainty = np.full(3, np.nan)
    uncertainty[fit_dims] = np.sqrt(np.clip(np.diag(covariance_fit), 0.0, np.inf))

    return location, uncertainty


@util.timeit()
def gaussian_filter(coalescence_map: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    Smooth the 2- or 3-D marginalised coalescence map using an appropriate Gaussian
    function to enable a better Gaussian fit to the data to be calculated.

    Parameters
    ----------
    coalescence_map:
        Marginalised coalescence map.
    sigma:
        Sigma value (in grid nodes) for the Gaussian filter function; larger sigma
        leads to more aggressive (long wavelength) smoothing.

    Returns
    -------
    smoothed_map:
        Gaussian smoothed 3-D coalescence map.

    """

    # Construct dimension-appropriate Gaussian filter
    depth_constrained = coalescence_map.shape[2] == 1
    if depth_constrained:
        map_to_smooth = coalescence_map[:, :, 0]
        gaussian_filter = gaussian_2d(*map_to_smooth.shape, sigma)
    else:
        map_to_smooth = coalescence_map
        gaussian_filter = gaussian_3d(*map_to_smooth.shape, sigma)
    reverse_slices = (REV_SLICE,) * map_to_smooth.ndim

    # Convolve coalescence_map and Gaussian filter
    smoothed_map = fftconvolve(map_to_smooth, gaussian_filter, mode="same")
    # Mirror and convolve again (to avoid "phase-shift")
    smoothed_map = smoothed_map[reverse_slices] / np.nanmax(smoothed_map)
    smoothed_map = fftconvolve(smoothed_map, gaussian_filter, mode="same")
    # Final mirror and normalise
    smoothed_map = smoothed_map[reverse_slices] / np.nanmax(smoothed_map)

    if depth_constrained:
        smoothed_map = smoothed_map[:, :, np.newaxis]

    return smoothed_map


def _mask_nd(shape: np.ndarray, centre: np.ndarray, window: int) -> np.ndarray:
    """
    Creates a mask that can be applied to a 2- or 3-D grid.

    Parameters
    ----------
    shape:
        Shape of grid.
    centre:
        Location of node around which to mask.
    window:
        Size of window around node to mask - window of grid nodes is
        +/-(win-1)//2 in x, y and z.

    Returns
    -------
    mask:
        Masking array.

    """

    half_window = (window - 1) // 2
    grids = np.indices(shape)

    centre = np.asarray(centre)
    centre = centre.reshape((-1,) + (1,) * len(shape))

    return np.all(np.abs(grids - centre) <= half_window, axis=0)


def gaussian_2d(nx: np.ndarray, ny: np.ndarray, sigma: float | int) -> Callable:
    """
    Create a 2-dimensional Gaussian function.

    Parameters
    ----------
    nx:
        Array of x values.
    ny:
        Array of y values.
    sigma:
        Sigma (width of gaussian in all directions).

    Returns
    -------
    f:
        2-dimensional Gaussian function

    """

    nx2 = (nx - 1) / 2
    ny2 = (ny - 1) / 2
    x = np.linspace(-nx2, nx2, nx)
    y = np.linspace(-ny2, ny2, ny)
    ix, iy = np.meshgrid(x, y, indexing="ij")

    if np.isscalar(sigma):
        sigma = np.repeat(sigma, 2)
    sx, sy = sigma

    f = np.exp(-(ix * ix) / (2 * sx * sx) - (iy * iy) / (2 * sy * sy))

    return f


def gaussian_3d(
    nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, sigma: float | int
) -> Callable:
    """
    Create a 3-dimensional Gaussian function.

    Parameters
    ----------
    nx:
        Array of x values.
    ny:
        Array of y values.
    nz:
        Array of z values.
    sigma:
        Sigma (width of gaussian in all directions).

    Returns
    -------
    f:
        3-dimensional Gaussian function

    """

    nx2 = (nx - 1) / 2
    ny2 = (ny - 1) / 2
    nz2 = (nz - 1) / 2
    x = np.linspace(-nx2, nx2, nx)
    y = np.linspace(-ny2, ny2, ny)
    z = np.linspace(-nz2, nz2, nz)
    ix, iy, iz = np.meshgrid(x, y, z, indexing="ij")

    if np.isscalar(sigma):
        sigma = np.repeat(sigma, 3)
    sx, sy, sz = sigma

    f = np.exp(
        -(ix * ix) / (2 * sx * sx)
        - (iy * iy) / (2 * sy * sy)
        - (iz * iz) / (2 * sz * sz)
    )

    return f


def build_gaussian_design_matrix(
    coords: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, int | None, int | None]]]:
    """
    Build the linear least-squares design matrix for an N-D Gaussian fit.

    The Gaussian is fitted by taking the negative logarithm of the coalescence values,
    so the fitted surface is a quadratic polynomial:

        -log(A) = x.T @ M @ x + h.T @ x + c

    Parameters
    ----------
    coords:
        Coordinate offsets from the coalescence maximum, with shape (n_dims, n_points).
        For example, in 3-D this contains the centred x, y, z coordinates of all
        selected grid nodes.

    Returns
    -------
    design_matrix:
        Matrix with one row per polynomial term and one column per selected grid node.
    terms:
        Metadata describing each row of the design matrix. This is used later to
        reconstruct the quadratic matrix, linear vector, and constant term.

    """

    n_dims, n_points = coords.shape

    columns, terms = [], []

    # Squared terms
    for i in range(n_dims):
        columns.append(coords[i] ** 2)
        terms.append(("square", i, i))

    # Cross terms
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            columns.append(coords[i] * coords[j])
            terms.append(("cross", i, j))

    # Linear terms
    for i in range(n_dims):
        columns.append(coords[i])
        terms.append(("linear", i, None))

    # Constant offset
    columns.append(np.ones(n_points))
    terms.append(("constant", None, None))

    return np.vstack(columns), terms


def unpack_gaussian_quadratic_terms(
    parameters: np.ndarray,
    terms: list[tuple[str, int | None, int | None]],
    n_dims: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert fitted polynomial coefficients into the compact quadratic Gaussian form:

        y(x) = x.T @ M @ x + h.T @ x + c

    where:

        M is the symmetric quadratic matrix,
        h is the linear coefficient vector,
        c is the constant offset.

    Parameters
    ----------
    parameters:
        Fitted least-squares coefficients, in the same order as terms.
    terms:
        Metadata returned by _gaussian_design_matrix.
    n_dims:
        Number of active dimensions in the Gaussian fit.

    Returns
    -------
    quadratic_matrix:
        Symmetric matrix, M.
    linear_vector:
        Vector, h.
    constant:
        Scalar offset, c.

    """

    quadratic_matrix = np.zeros((n_dims, n_dims))
    linear_vector = np.zeros(n_dims)
    constant = np.nan

    for parameter, (kind, i, j) in zip(parameters, terms):
        if kind == "square":
            quadratic_matrix[i, i] = parameter

        elif kind == "cross":
            quadratic_matrix[i, j] = parameter / 2.0
            quadratic_matrix[j, i] = parameter / 2.0

        elif kind == "linear":
            linear_vector[i] = parameter

        elif kind == "constant":
            constant = parameter

    return quadratic_matrix, linear_vector, constant


def embed_matrix(matrix: np.ndarray, fit_dims: list[int]) -> np.ndarray:
    """Utility to embed a 2-D matrix back into 3-D space."""

    full_matrix = np.full((3, 3), np.nan)
    full_matrix[np.ix_(fit_dims, fit_dims)] = matrix

    return full_matrix

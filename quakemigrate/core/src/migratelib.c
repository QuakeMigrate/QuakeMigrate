/*
 * =============================================================================
 *
 *       Filename:  migratelib.c
 *
 *        Purpose:  Routines for computing the 4-D coalescence function and
 *                  determining the maximum values.
 *
 *      Copyright:  2020-2023, QuakeMigrate developers.
 *        License:  GNU General Public License, Version 3
 *                  (https://www.gnu.org/licenses/gpl-3.0.html)
 *
 * =============================================================================
 */

#include "qmlib.h"

/*
 * Function: migrate
 * -----------------
 * Compute time series of the coalescence function in a 3-D volume by migrating
 * and stacking onset functions, using the geometric mean as the stacking
 * operator.
 *
 * Arguments
 * ---------
 * onsets: Pointer to input array containing onset functions for each seismic
 *         phase and station.
 * lookup_tables: Pointer to input array containing seismic phase traveltimes,
 *                converted to an integer multiple of the sampling rate.
 * map4d: Pointer to output array for 4-D coalescence map.
 * fsmp: Index of first sample in array from which to scan.
 * lsmp: Index of last sample in array up to which to scan.
 * n_samples: Total number of samples over which to scan.
 * n_stations: Number of stations available.
 * available: Number of onset functions available.
 * n_nodes: Total number of nodes in the 3-D grid.
 * threads: Number of threads with which to scan.
 */
void
migrate(double * onsets, int32_t * lookup_tables, double * map4d, int32_t fsmp,
  int32_t lsmp, int32_t n_samples, int32_t n_stations, int32_t available,
  int64_t n_nodes, int64_t threads) {

  double * stack;
  int32_t station, sample, * traveltimes, traveltime;
  int64_t node;

  #pragma omp parallel\
  for private(station, stack, sample, traveltimes, traveltime) num_threads(threads)
  for (node = 0; node < n_nodes; ++node) {
    stack = & map4d[node * (int64_t) n_samples];
    traveltimes = & lookup_tables[node * (int64_t) n_stations];
    for (station = 0; station < n_stations; ++station) {
      traveltime = MAX(0, traveltimes[station]);
      for (sample = 0; sample < n_samples; ++sample) {
        stack[sample] += onsets[station * (fsmp + lsmp + n_samples) + \
          traveltime + fsmp + sample];
      }
    }
    for (sample = 0; sample < n_samples; ++sample) {
      stack[sample] = exp(stack[sample] / available);
    }
  }
}

/*
 * Function: find_max_coa
 * ----------------------
 * Find the time series of maximum and normalised maximum coalescence values,
 * and their corresponding grid indices.
 *
 * Arguments
 * ---------
 * map4d: Pointer to input array containing 4-D coalescence map.
 * max_coa: Pointer to output array for maximum coalescence time series.
 * max_norm_coa: Pointer to output array for normalised maximum coalescence
 *               time series
 * max_coa_idx: Point to output array for corresponding indices for the maximum
 *              coalescence time series.
 * n_samples: Total number of samples over which to scan.
 * n_nodes: Total number of nodes in the 3-D grid.
 * threads: Number of threads with which to scan.
 */
void
find_max_coa(double * map4d, double * max_coa, double * max_norm_coa,
  int64_t * max_coa_idx, int32_t n_samples, int64_t n_nodes, int64_t threads) {

  double current, max, sum;
  int32_t sample;
  int64_t node, idx;

  #pragma omp parallel\
  for private(node, current, idx, max, sum) num_threads(threads)
  for (sample = 0; sample < n_samples; ++sample) {
    max = map4d[sample];
    idx = 0;
    sum = map4d[sample];
    for (node = 1; node < n_nodes; ++node) {
      current = map4d[node * (int64_t) n_samples + (int64_t) sample];
      sum += current;
      if (current > max) {
        max = current;
        idx = node;
      }
    }
    max_coa[sample] = max;
    max_norm_coa[sample] = max * n_nodes / sum;
    max_coa_idx[sample] = idx;
  }
}

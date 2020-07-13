/*
 * =============================================================================
 *
 *       Filename:  quakemigrate.c
 *
 *        Purpose:  Routines for computing the 4-D coalescence function and
 *                  determining the maximum values.
 *
 *        Created:  15/05/2020
 *       Revision:  none
 *       Compiler:  gcc/clang
 *
 *         Author:  QuakeMigrate developers.
 *   Organization:  QuakeMigrate
 *      Copyright:  QuakeMigrate developers.
 *
 * =============================================================================
 */
#include <stdint.h>
#include <math.h>

#ifndef _OPENMP
    /* Generate a compiler error to stop the build */
    mustLinkOpenMP
#endif

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(_GCC)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    #define EXPORT
    #define IMPORT
    // #pragma warning Unknown dynamic link import/export semantics.
#endif

/* Macros for min/max. */
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

EXPORT void
migrate(double *sigPt, int32_t *indPt, double *mapPt, int32_t fSamp,
        int32_t lSamp, int32_t nSamps, int32_t nStations, int32_t avail,
        int64_t nNodes, int64_t threads)
{
    /*
    Purpose: compute time series of the coalescence function in a 3-D volume
             by migrating and stacking onset functions.

    Args:
      sigPt: Pointer to array containing onset functions for each seismic phase
      indPt: Pointer to array containing seismic phase traveltimes, converted
             to an integer multiple of the sampling rate
      mapPt: Pointer to array in which to output 4-D coalescence map
      fSamp: Index of first sample in array from which to scan.
      lSamp: Index of last sample in array up to which to scan.
      nSamps: Total number of samples over which to scan.
      nStations: Number of stations available.
      avail: Number of onset functions available.
      nNodes: Total number of nodes in the 3-D grid.
      threads: Number of threads with which to scan.
    */
    double  *stnPt, *stkPt, *eStkPt;
    int32_t station, t, *ttpPt, ttp;
    int64_t node;

    #pragma omp parallel for \
    private(eStkPt, station, stkPt, stnPt, t, ttp, ttpPt) \
    num_threads(threads)
    for (node=0; node<nNodes; node++) {
        stkPt = &mapPt[node * (int64_t) nSamps];
        eStkPt = &mapPt[node * (int64_t) nSamps];
        ttpPt = &indPt[node * (int64_t) nStations];
        for(station=0; station<nStations; station++) {
            ttp = MAX(0, ttpPt[station]);
            stnPt = &sigPt[station*(fSamp + lSamp + nSamps) + ttp + fSamp];
            for(t=0; t<nSamps; t++){
                stkPt[t] += stnPt[t];
            }
        }
        for(t=0; t<nSamps; t++) {
            eStkPt[t] = exp(stkPt[t] / avail);
        }
    }
}


EXPORT void
find_max_coa(double *mapPt, double *snrPt, double *nsnrPt, int64_t *indPt,
             int32_t nSamps, int64_t nNodes, int64_t threads)
{
    /*
    Purpose: find the time series of maximum coalescence, normalised maximum
             coalescence, and the corresponding indices.

    Args:
      mapPt: Pointer to array containing 4-D coalescence map
      snrPt: Pointer to array in which to output maximum coalescence time
             series
      nsnrPt: Pointer to array in which to output normalised maximum
              coalescence time series
      indPt: Pointer to array in which to output the corresponding indices for
             the maximum coalescence time series
      nSamps: Total number of samples over which to scan.
      nNodes: Total number of nodes in the 3-D grid.
      threads: Number of threads with which to scan.
    */
    double  curVal, maxVal, sumVal;
    int32_t t;
    int64_t node, idx;

    #pragma omp parallel for \
    private(node, curVal, idx, maxVal, sumVal) \
    num_threads(threads)
    for (t=0; t<nSamps; t++) {
        maxVal = mapPt[t];
        idx = 0;
        sumVal = mapPt[t];
        for (node=1; node<nNodes; node++) {
            curVal = mapPt[node * (int64_t) nSamps + (int64_t) t];
            sumVal += curVal;
            if (curVal > maxVal) {
                maxVal = curVal;
                idx = node;
            }
        }
        snrPt[t] = maxVal;
        nsnrPt[t] = maxVal * nNodes / sumVal;
        indPt[t] = idx;
    }
}
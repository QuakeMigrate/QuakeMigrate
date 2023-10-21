/*
 * =============================================================================
 *
 *       Filename:  qmlib.h
 *
 *        Purpose:  Header file to bring together definitions used in the
 *                  quakemigrate.c library.
 *
 *      Copyright:  2020-2023, QuakeMigrate developers.
 *        License:  GNU General Public License, Version 3
 *                  (https://www.gnu.org/licenses/gpl-3.0.html)
 *
 * =============================================================================
 */

#include <stdint.h>
#include <math.h>

#ifndef _OPENMP
    /* Generate a compiler error to stop the build */
    mustLinkOpenMP
#endif

/* Macros for min/max. */
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void migrate(double*, int32_t*, double*, int32_t, int32_t, int32_t, int32_t,
             int32_t, int64_t, int64_t);

void find_max_coa(double*, double*, double*, int64_t*, int32_t, int64_t,
                  int64_t);

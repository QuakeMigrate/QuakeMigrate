#include <stdint.h>
#include <math.h>

#ifndef _OPENMP
    #define STRING2(x) #x
    #define STRING(x) STRING2(x)
    #pragma message (__FILE__ "(" STRING(__LINE__) "): error: This module should be compiled with /openmp on the command line")

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

EXPORT void migrate(double *sigPt, int32_t *indPt, double *mapPt, int32_t fsmp, int32_t lsmp, int32_t nsamp, int32_t nstation, int32_t avail, int64_t ncell, int64_t threads)
{
    double  *stnPt, *stkPt, *eStkPt;
    int32_t *ttpPt;
    int32_t ttp;
    int32_t to, tm, st;
    int64_t cell;

    #pragma omp parallel for num_threads(threads)
    for (cell=0; cell<ncell; cell++)
    {
        stkPt = &mapPt[cell * (int64_t) nsamp];
        eStkPt = &mapPt[cell * (int64_t) nsamp];
        ttpPt = &indPt[cell * (int64_t) nstation];
        for(st=0; st<nstation; st++)
        {
            ttp   = MAX(0,ttpPt[st]);
            stnPt = &sigPt[st*(fsmp + lsmp + nsamp) + ttp + fsmp];
            for(tm=0; tm<nsamp; tm++)
            {
                stkPt[tm] += stnPt[tm];
            }
        }
        for(tm=0; tm<nsamp; tm++)
        {
            eStkPt[tm] = exp(stkPt[tm] / avail);
        }
    }
}


EXPORT void find_max_coa(double *mapPt, double *snrPt, double *nsnrPt, int64_t *indPt, int32_t fsmp, int32_t lsmp, int32_t nsamp, int64_t ncell, int64_t threads)
{
    double  mv, cv, sm;
    int32_t tm;
    int64_t cell, ix;

    #pragma omp parallel for num_threads(threads)
    for (tm=fsmp; tm<lsmp; tm++)
    {
        mv = 0.0;
        ix = 0;
        sm = 0.0;
        for (cell=0; cell<ncell; cell++)
        {
            cv = mapPt[cell * (int64_t) nsamp + (int64_t) tm];
            sm += cv;
            if (cv > mv)
            {
                mv = cv;
                ix = cell;
            }
        }
        snrPt[tm] = mv;
        nsnrPt[tm] = mv * ncell / sm;
        indPt[tm] = ix;
    }
}

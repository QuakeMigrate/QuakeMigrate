
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

EXPORT void scan4d(double *sigPt, int32_t *indPt, double *mapPt, int32_t fsmp, int32_t lsmp, int32_t nsamp, int32_t nstation, int64_t ncell, int64_t threads, int64_t samplingRate, int8_t modeT, int64_t dropTime)
{
    double  *stnPt, *stkPt;
    int32_t *ttpPt;
    int32_t ttp, tend;
    int32_t to, tm, st;
    int64_t cell;
    double attenuate = 1.0;
    double dropTime_S = dropTime * 1.68;
    int trueNstations = (int) nstation/2;

    /* omp_set_num_threads(threads); */

    /* shared(mapPt) */
    #pragma omp parallel for private(cell,tm,st,stnPt,stkPt,ttpPt,ttp,tend) num_threads(threads)
    for (cell=0; cell<ncell; cell++)
    {
        stkPt = &mapPt[cell * (int64_t) nsamp];
        ttpPt = &indPt[cell * (int64_t) nstation]; //first half of stations are P, second half are S
        for(st=0; st<trueNstations; st++)
        {
            ttp   = MAX(0,ttpPt[st]);
            stnPt = &sigPt[st*(fsmp + lsmp + nsamp) + ttp + fsmp];
            if (modeT == 0)
            {
                attenuate = 1;
                    if (((double)(ttp)/(double)samplingRate) > (double)dropTime)
                    {
                        attenuate = 0;
                    }
            }
            if (modeT == 1)
            {
                attenuate = exp( -0.5 * ((double)(ttp) / (double)(samplingRate)) / (double)dropTime );
            }
            for(tm=0; tm<nsamp; tm++)
                stkPt[tm] += attenuate * stnPt[tm];
        }
        for(st=trueNstations; st<nstation; st++)
        {
            ttp   = MAX(0,ttpPt[st]);
            stnPt = &sigPt[st*(fsmp + lsmp + nsamp) + ttp + fsmp];
            if (modeT == 0)
            {
                attenuate = 1;
                if (((double)(ttp)/(double)samplingRate) > (double)dropTime_S)
                {
                    attenuate = 0;
                }
            }
            if (modeT == 1)
            {
                attenuate = exp( -0.5 * ((double)(ttp) / (double)(samplingRate)) / (double)dropTime );
            }
            for(tm=0; tm<nsamp; tm++)
                stkPt[tm] += attenuate * stnPt[tm];
        }
    }
}


EXPORT void detect4d(double *mapPt, double *snrPt, int64_t *indPt, int32_t fsmp, int32_t lsmp, int32_t nsamp, int64_t ncell, int64_t threads)
{
    double  mv, cv;
    int32_t tm;
    int64_t cell, ix;

    /* omp_set_num_threads(threads); */

    /* stack data.... */
    /* shared(mapPt) */
    #pragma omp parallel for private(cell,tm,mv,ix,cv) num_threads(threads)
    for (tm=fsmp; tm<lsmp; tm++)
    {
        mv = 0.0;
        ix = 0;
        for (cell=0; cell<ncell; cell++)
        {
            cv = mapPt[cell * (int64_t) nsamp + (int64_t) tm];
            if (cv > mv)
            {
                mv = cv;
                ix = cell;
            }
        }
        snrPt[tm] = mv;
        indPt[tm] = ix;
    }
}

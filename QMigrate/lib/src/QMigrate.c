
#include <stdint.h>

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
#define	MIN(a,b) (((a)<(b))?(a):(b))
#define	MAX(a,b) (((a)>(b))?(a):(b))


EXPORT void scan4d(double *sigPt, int32_t *indPt, double *mapPt, int32_t fsmp, int32_t lsmp, int32_t nsamp, int32_t nstation, int64_t ncell, int64_t threads)
{
	double	 *stnPt, *stkPt;
	int32_t  *ttpPt;
	int32_t  ttp, tend;
    int32_t  to, tm, st;
	int64_t  cell;

	omp_set_num_threads(threads);

	#pragma omp parallel for private(cell,tm,st,stnPt,stkPt,ttpPt,ttp,tend) /* shared(mapPt) */
	for (cell=0; cell<ncell; cell++)
	{
		stkPt = &mapPt[cell * (int64_t) nsamp];
		ttpPt = &indPt[cell * (int64_t) nstation];
		// for(tm=0; tm<lsmp-to; tm++)
		// 	stkPt[tm]  = 0.0;
		for(st=0; st<nstation; st++)
		{
			ttp     = MAX(0,ttpPt[st]);
			stnPt   = &sigPt[st*(fsmp + lsmp + nsamp) + ttp + fsmp];
			for(tm=0; tm<nsamp; tm++)
				stkPt[tm] += stnPt[tm];
		}
	}
}


EXPORT void detect4d(double *mapPt, double *snrPt, int64_t *indPt, int32_t fsmp, int32_t lsmp, int32_t nsamp, int64_t ncell, int64_t threads)
{
	double	   mv, cv;
	int32_t  tm;
	int64_t  cell, ix;

	/* stack data.... */

	omp_set_num_threads(threads);

	#pragma omp parallel for private(cell,tm,mv,ix,cv) /* shared(mapPt) */
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

// EXPORT void detect4d_t(double *mapPt, double *snrPt, int64_t *indPt, int32_t fsmp, int32_t lsmp, int32_t nsamp, int64_t ncell, int64_t threads)
// {
// 	double	   *rPt, mv, cv;
// 	int32_t  tm;
// 	int64_t  cell, ix;

// 	/* stack data.... */

// 	omp_set_num_threads(threads);

// 	#pragma omp parallel for private(cell,tm,mv,ix,cv,rPt)  shared(mapPt) 
// 	for (tm=fsmp; tm<lsmp; tm++)
// 	{
// 	    mv  = 0.0;
// 	    ix  = 0;
// 	    rPt = &mapPt[ncell * (int64_t) tm];
// 	    for (cell=0; cell<ncell; cell++)
// 	    {
// 	        cv = rPt[cell];
// 	        if ( cv > mv)
// 	        {
// 	            mv = cv;
// 	            ix = cell;
// 	        }
// 	    }
// 	    snrPt[tm] = mv;
// 	    indPt[tm] = ix;
// 	}
// }

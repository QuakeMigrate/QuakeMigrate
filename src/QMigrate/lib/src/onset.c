
#include <math.h>

#ifndef _OPENMP
  #define STRING2(x) #x
  #define STRING(x) STRING2(x)
  #pragma message (__FILE__ "(" STRING(__LINE__) "): error: This module should be compiled with /openmp on the command line")

  /* Generate a compiler error to stop the build as the above message doesn't stop the build when building in Matlab. */
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

EXPORT void onset(double *envPt, int nsmp, int stw, int ltw, int gap, double *onsPt)
{
	int     index, ii;
    int     gap2;
	double	ssw, slw, scl;
	double	*sswPt, *elwPt, *eswPt;

	gap2 = gap >> 1;
	ssw=0.0;slw=0.0;scl=(double) ltw/stw;
	index=nsmp-ltw-stw-2*gap2;
	onsPt=&onsPt[ltw+gap2];
	sswPt=&envPt[ltw+gap2+gap2+1];

	for (ii=0; ii < ltw; ii++) { slw+=envPt[ii]; }
	for (ii=0; ii < stw; ii++) { ssw+=sswPt[ii]; }
	onsPt[0] = (ssw/slw)*scl;
	elwPt=&envPt[ltw-1];
	eswPt=&sswPt[stw-1];
	for (ii=1; ii < index; ii++) {
		slw=slw+elwPt[ii]-envPt[ii-1];
		ssw=ssw+eswPt[ii]-sswPt[ii-1];
		if (slw > 0)
			onsPt[ii] = (ssw/slw)*scl;
		else
			onsPt[ii] = 0.0;
	}
	
}

EXPORT void onset_mp(double *dataPt, int ntr, int nsamp, int swin, int lwin, int gap, double *resultPt)
{
	double *envPt, *onsPt;
	int tr;

	#pragma omp parallel for private(tr,envPt,onsPt)
	for (tr=0; tr<ntr; tr++)
	{
		envPt = &dataPt[tr*nsamp];
		onsPt = &resultPt[tr*nsamp];
		onset(envPt, nsamp, swin, lwin, gap, onsPt);
	}
	
}

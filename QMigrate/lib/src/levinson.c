
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


EXPORT int nlevinson(const double* in, int nsamp, double* acoeff, double* tmp)
{
	int i, j;
    double	err, norm;
	int ret = 0;
    
    acoeff[0]=1/in[0];
  
    for (j=1; (j<nsamp); j++) {
        err=0;
        for (i=0; (i<j); i++) {
            err=err+acoeff[i]*in[j-i];
        }
        norm=1/(1-err*err);
        for (i=0; (i<j+1); i++) {
            tmp[i]=norm*(acoeff[i]-err*acoeff[j-i]);
        }
        for (i=0; (i<j+1); i++) {
            acoeff[i]=tmp[i];
        }
    }
    return ret;
}

 
EXPORT int nlevinson_mp(const double* in, int nchan, int nsamp, double* acoeff, double* tmp)
{
        int smp, ch;
        int ret = 0;

		#pragma omp parallel for private(ch, smp)
        for (ch = 0; ch < nchan; ch++) {
			smp = ch*nsamp;
			nlevinson(&in[smp], nsamp, &acoeff[smp], &tmp[smp]);
        }

        return ret;
}


EXPORT int levinson(const double* in, int order, double* acoeff, double* err,
             double* kcoeff, double* tmp)
{
        int i, j;
        double  acc;
        int ret = 0;

        /* order 0 */
        acoeff[0] = (double)1.0;
        *err = in[0];

        /* order >= 1 */
        for (i = 1; i <= order; ++i) {
                acc = in[i];
                for ( j = 1; j <= i-1; ++j) {
                        acc += acoeff[j]*in[i-j];
                }
                kcoeff[i-1] = -acc/(*err);
                acoeff[i] = kcoeff[i-1];

                for (j = 0; j < order; ++j) {
                        tmp[j] = acoeff[j];
                }

                for (j = 1; j < i; ++j) {
                        acoeff[j] += kcoeff[i-1]*tmp[i-j];
                }
                *err *= (1-kcoeff[i-1]*kcoeff[i-1]);
        }

        return ret;
}

EXPORT int levinson_mp(const double* in, int nchan, int nsamp, int order, double* acoeff, double* err,
             double* kcoeff, double* tmp)
{
        int ch, dn, an, kn;
        int ret = 0;

		#pragma omp parallel for private(ch, dn, an, kn)
        for (ch = 0; ch < nchan; ch++) {
			dn = ch*nsamp;
			an = ch*(order+1);
			kn = ch*order;
			levinson(&in[dn], order, &acoeff[an], &err[ch],
                     &kcoeff[kn], &tmp[kn]);
        }

        return ret;
}


import numpy as np
from scipy.special import erf
from scipy.stats import norm

def gaussian_time_prob(t, pick_time, time_error): 
    """
    Gaussian time probability function 
    """
    return norm.pdf(t, pick_time, time_error)

def amp_prob(pol, x, noise_error):
    """
    Amplitude probability function (see Pugh, D J, White, R S  and Christie, P A F, 2015. Automatic Bayesian polarity determination, GJI, submitted).
    """
    p_x = 0.5 * (1 + erf(x[0:-2] * pol[0:-1] / \
            (np.sqrt(2) * noise_error)))
    return np.append(p_x,0)  

def prob_polarity(trace, time_pdf, noise_error):    
    """
    Calculates the polarity probability following the approach described in Pugh, D J, White, R S  and Christie, P A F, 2015. Automatic Bayesian polarity determination, GJI, submitted.
    The time PDF is  generated separately to allow for different arrival time PDFs to be used. The Gaussian time PDF can be generated using autopol.probability.time_pdf function or otherwise, 
    as all that is expected is a numpy array.

    Returns
        pPositive - probability of a positive polarity arrival.
        pNegative - probability of a negative polarity arrival.
        pPositive(t) - the time series of the positive polarity probability distribution (numpy array).
        pNegative(t) - the time series of the negative polarity probability distribution (numpy array).
        pol(t) - the time series of the polarity measurement (numpy array).
        delta_x(t) - the time series of the amplitude change between polarity extrema (numpy array).
        pX(t) - the time series of the positive polarity amplitude estimate (numpy array).
        pT(t) - the time series of the pick time PDF (numpy array).
    """
    #trace and time are expected to be numpy array
    assert len(time_pdf) == len(trace) - 1

    #Get np.difference between values
    dx = np.diff(trace)
    #np.diff is a[n+1]-a[n] corresponds in effect to n+0.5
    #Get second derivative
    d2x = np.diff(dx)
    #d2x[n] therefore corresponds to trace[n+1] ==> prepend a zero to match n to n
    d2x = np.insert(d2x, 0, 0.)
    #to evaluate for turning point need to check np.diff[n]*np.diff[n-1] to see if sign changes - this means that a turning point is at n if negative.
    turning_points = (np.insert(dx, 0, dx[0]) * np.append(dx, dx[-1])) <= 0 # logical array
    #To get Polarity for that position with a turning point need to multiply d2x into that...

    pol = -d2x * turning_points[0:-1]
    zeros = (pol==0.)
    pol[~zeros] = pol[~zeros] / np.abs(pol[~zeros])
    zeros = np.where(pol==0)[0]
    non_zeros = np.where(pol)[0]
    for i in zeros:
        try:
            pol[i] = pol[non_zeros[np.where(non_zeros>i)[0][0]]]#Sets pol values to be consistent with prev non-zero
        except Exception:
            pass
    #Get time prob

    #Add 0 to start of shifts so that delta is between current turning point and previous...
    dx = abs(np.diff(np.insert(trace[turning_points], 0, 0)))
    zeros = np.where(turning_points==0)[0]
    non_zeros = np.where(turning_points)[0]
    delta_x = np.zeros(len(trace))
    for i, ind in enumerate(non_zeros):
        try:
            delta_x[ind]=dx[i]
        except:
            pass
    for i in zeros:
        try:
            delta_x[i]=delta_x[non_zeros[np.where(non_zeros>i)[0][0]]]
        except:
            pass
    p_x = amp_prob(pol, delta_x, noise_error)
    pp_t = p_x * time_pdf
    pn_t = amp_prob(-pol, delta_x, noise_error) * time_pdf
    #marginalise with respect to time (same length so dt irrelevant as normalising)
    pp = np.sum(pp_t)
    pn = np.sum(pn_t)
    #normalise 
    n = pp + pn  
    pp_t = pp_t / n
    pn_t = pn_t / n
    pp = pp / n
    pn = pn / n
    return pp, pn, pp_t, pn_t, pol, delta_x[0:-1], p_x, time_pdf

def polarity_probability(trace, time, pick_time, 
            noise_error, time_error, 
            time_pdf='gaussian', *args, **kwargs):
    """

    Other time pdfs can be used by submitting the time_pdf argument as 
    a numpy array, and leaving the time argument as an empty list 

    """
    try:
        a = trace.data
    except:
        a = trace
    if np.max(a) == np.min(a):
        raise ValueError('all data are equal to zero')
    if not type(time) == type(np.array([])):
        time = np.array(time)
    #for array time_pdf 
    if time_pdf == 'gaussian':
        time_pdf = gaussian_time_prob(time, pick_time, time_error)[:-1]
    elif type(time_pdf) in [type(np.array([])),type(np.matrix([]))]:
        pass
    else:
        raise AttributeError(time_pdf, 'Time pdf is not possible')
    return prob_polarity(a, time_pdf, noise_error)


def obspy_trace_fn(trace, pick_time, 
                    noise_error, time_error, 
                    time_pdf='gaussian', return_all=False):
    """
    Function to act on obspy.core.trace.Trace class and return the 
    polarity probabilities
    """
    t = trace.times()
    pick_time = pick_time - trace.stats.starttime

    pp, pn, ppt, pnt, pol, delta_x, p_x, p_t = polarity_probability(trace, t, pick_time,
                                                            noise_error, time_error, time_pdf)
    #Handle traces
    trace.stats['pPositive'] = pp
    trace.stats['pNegative'] = pn
    if return_all:
        return trace,ppt,pnt,pol,delta_x,p_x,p_t
    return pp, pn
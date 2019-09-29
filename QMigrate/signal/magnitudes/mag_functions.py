import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
import pandas as pds

def calculate_mean_magnitude(obj, params):
    """
    params : dict
        A dictionary containing various parameters needed for the calculation.

    It may contain these keys:
        'trace_filter' : reg_expr : a regular expression for selecting particular channels 
            to use in the average 
        'dist_filter' : float : only use reported magnitudes from distances less than the 
            value
        'noise_filter' : float : use only those channels where the amplitude is more than 
            the parameter multiplied by the pre-signal noise
        'use_only_picked' : bool : use only channels which have been picked by the 
            autopicker
        'weighted' : bool : calculate a weighted average
    """

    if 'trace_filter' in params and params['trace_filter']:
        obj = obj.filter(regex=params['trace_filter'], axis=0)
    
    if 'dist_filter' in params and params['dist_filter']:
        if params['use_hyp_distance']:
            dist = np.sqrt(np.square(obj['epi_dist'].values) + np.square(obj['depth'].values))
        else:
            dist = obj['epi_dist'].values
        obj = obj[dist <= params['dist_filter']]
    
    if 'use_only_picked' in params and params['use_only_picked']:
        obj = obj[obj['is_picked']]
    
    if 'noise_filter' in params and params['noise_filter']:
        feat = obj[params['amplitude_feature']].values
        noise = obj[params['Error']].values
        obj = obj[feat >= noise * params['noise_filter']]
    
    if len(obj) == 0:
        return np.nan, np.nan
    
    mag = obj['Magnitude'].values

    if 'weighted' in params and params['weighted']:
        wght = obj['Magnitude_err']
    else:
        wght = np.ones_like(mag)
    
    mean_mag = (mag * wght) / np.sum(wght)
    mean_mag_error = np.sqrt((np.sum((mag - mean_mag) * wght) / np.sum(wght)) ** 2.)

    return mean_mag, mean_mag_error

def calculate_magnitudes(obj, params):

    """
    Calculate the magnitude from the output of Quake Migrate

    Inputs
    obj : pandas DataFrame 
        A pandas Dataframe object containing the trace ids, amplitude 
    measurements, epicentral distance and depth

    params : dict
        A dictionary containing various parameters needed for the calculation.
    It must have these keys:
        'station_corrections' : dictionary of id, correction pairs, missing stations
            don't matter. use a blank dictionary if none exist
        'amplitude feature' : which amplitude feature to do the calculation on.
            Normally 'S_amp' makes sense. This should be the full peak-to-trough
            amplitude
        'use_hyp_distance' : make true if you want to use hypocentral distance 
            rather than epicentral distance
        'A0' : which A0 attenuation correction to use. See logA0 function for options
            or pass the function directly (i.e. lambda)
    """

    try:
        stcorr = params['station_corrections']
    except KeyError:
        stcorr = {}

    idx = obj.index
    amp = obj[params['amplitude_feature']].values
    amp_err = obj['Error'].values
    if params['use_hyp_distance']:
        dist = np.sqrt(np.square(obj['epi_dist'].values) + np.square(obj['depth'].values))
    else:
        dist = obj['epi_dist'].values

    obs['Magnitude'] = calc_mag(idx, amp, dist, obj['A0'], stcorr)
    mag_error_upper = calc_mag(idx, amp + amp_err, dist, obj['A0'], stcorr)
    mag_error_lower = calc_mag(idx, amp - amp_err, dist, obj['A0'], stcorr)
    obs['Magnitude_err'] = mag_error_upper - mag_error_lower

    return obs

def calc_mag(trace_ids, amplitudes,
        dist, A0_calib, station_corrections):

    corrs = np.array([station_corrections[tr_id] if tr_id in station_corrections.keys() else 0. for tr_id in trace_ids])

    if type(A0_calib) == 'function':
        att = A0_calib(dist)
    else:
        att = logA0(dist, A0_calib)

    return np.log10(amplitudes) + att + corrs

def logA0(dist, eqn):
    """
    A set of logA0 attenuation correction equations from the literature. Feel free to add more.

    Currently implemented:
        Keir et al. (2006) - Ethiopia - 'keir2006'
        Illsley-Kemp et al. (2017) - Danakil Depression, Afar - 'Danakil2017'
        Greenfield et al. (2018) - Askja, Iceland - 'Greenfield2018_askja'
        Greenfield et al. (2018) - Bardarbunga, Iceland - 'Greenfield2018_bardarbunga'
        Greenfield et al. (2018) - Askja & Bardarbunga, Iceland - 'Greenfield2018_comb'
        Hutton & Boore (1987) - Southern California - 'Hutton-Boore'
        Langston (1998) - Tanzania, East Africa - 'langston1998'
        Luckett et al (2018) - UK - 'UK'
    """
    if eqn == 'keir2006':
        return (1.196997 * np.log10(dist / 17.)) + (0.001066 * (dist - 17.)) + 2
    elif eqn == 'Danakil2017':
        return 1.274336 * np.log10(dist / 17.) - 0.000273 * (dist - 17.) + 2
    elif eqn == 'Greenfield2018_askja':
        return 1.4406 * np.log10(dist / 17.) + 0.003 * (dist - 17.) + 2
    elif eqn == 'Greenfield2018_bardarbunga':
        return 1.2534 * np.log10(dist / 17.) + 0.0032 * (dist - 17.) + 2
    elif eqn == 'Greenfield2018_comb':
        return 1.1999 * np.log10(dist / 17.) + 0.0016 * (dist - 17.) + 2
    elif eqn == 'Hutton-Boore':
        return 1.11 * np.log10(dist / 100.) + 0.00189 * (dist - 100.) + 3.
    elif eqn == 'langston1998':
        return 0.776 * np.log10(dist / 17.) + 0.000902 * (dist - 17) + 2.0
    elif eqn == 'UK':
        return 1.11 * np.log10(dist) + (0.00189 * dist) - 1.16 * np.exp(-0.2 * dist) - 2.09
    else:
        raise ValueError(eqn, 'is not a valid method')

def GR_mags(nevents, b_value, m_min):
    #Generates series of magnitudes according to GR distribution for a series of spikes
    #set-up random number generator
    rng = np.random.RandomState()
    rng.seed
    ms = m_min + rng.exponential(1. / (-b_value / np.log10(np.e)), nevents)
    return ms

def generate_synthetic_cat(nev, nsta, noise_level, 
                            xlim, ylim, zlim, 
                            magrange, stcorr_width, **kwargs):

    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim
    mmin, _ = magrange

    stloc = np.random.rand(nsta, 3)
    stloc[:, 0] = xmin + (stloc[:, 0] * (xmax - xmin))
    stloc[:, 1] = ymin + (stloc[:, 1] * (ymax - ymin))
    stloc[:, 2] = 0.

    eqloc = np.random.rand(nev, 3)
    eqloc[:, 0] = xmin + (eqloc[:, 0] * (xmax - xmin))
    eqloc[:, 1] = ymin + (eqloc[:, 1] * (ymax - ymin))
    eqloc[:, 2] = zmin + (eqloc[:, 2] * (zmax - zmin))

    # plt.figure()
    # plt.plot(stloc[:, 0], stloc[:, 1], 'k^')
    # plt.plot(eqloc[:, 0], eqloc[:, 1], 'ro')

    mags = GR_mags(nev, -1., mmin)
    n = np.random.normal(scale=stcorr_width, size=nsta)
    if np.all(n == 0):
        stcorr = dict(zip(range(nsta), n))
    else:
        n = (n / np.sum(n)) - (1. / nsta)
        stcorr = dict(zip(range(nsta), n))

    observations = {}
    i = 0
    while i < nev:
        amp = np.zeros((nsta * 2, 7))
        index = []
        uid = i
        j = 0
        while j < nsta:
            xdist = eqloc[i, 0] - stloc[j, 0]
            ydist = eqloc[i, 1] - stloc[j, 1]
            zdist = eqloc[i, 2] - stloc[j, 2]
            dist = np.sqrt(np.square(xdist) + np.square(ydist))

            noise = np.random.normal(scale=noise_level, size=2)

            loga0 = logA0(dist, 'keir2006')
            loga = mags[i] - loga0 - stcorr[j]
            a = np.power(10., loga)

            for k in range(2):
                amp[2*j + k, 0] = dist
                amp[2*j + k, 1] = np.abs(zdist)
                amp[2*j + k, 4] = (a + noise[k]) / 1000.
                while amp[2*j + k, 4] < 0.:
                    amp[2*j + k, 4] = (a + np.random.normal(scale=noise_level)) / 1000.
                amp[2*j + k, 6] = noise[k]
                index.append('.{:04d}..BH{:1d}'.format(j, k+1))
            j += 1
        amp = pds.DataFrame(amp, 
                columns=['epi_dist', 'depth', 
                        'P_amp', 'P_freq', 
                        'S_amp', 'S_freq', 
                        'Error'],
                index=index)  
        observations[uid] = amp
        i += 1
    
    return observations, mags, stcorr, (eqloc, stloc)

def invert_mag_scale(data, stcorr_only=False, **kwargs):
    """
    Inverts for the attenuation parameters from amplitude obseravtions

    inputs

    data : dict
        input data. Comprised of a dictionary of pandas DataFrame objects
        containing the amplitude data
    """

    nev = len(data.keys())
    uids = list(data.keys())
    uiddic = dict(zip(uids, range(nev)))

    comps = []
    for uid in uids:
        comps += list(data[uid].index.values)
    comps = list(set(comps))
    ncomps = len(comps)
    compdic = dict(zip(comps, np.linspace(0, ncomps - 1, ncomps) + nev))

    nobs = np.sum([len(data[uid]) for uid in uids]) + 1

    if stcorr_only:
        nvar = nev + ncomps
        n = kwargs['n']
        k = kwargs['k']
    else:
        nvar = nev + ncomps + 2
    
    A = sparse.lil_matrix((nobs, nvar))
    b = np.zeros(nobs)

    i = 0
    for uid in uids:
        d = data[uid]
        dist = d['epi_dist'].values
        amps = d['S_amp'].values
        tr_ids = d.index.values
        n = len(d)
        j = 0
        while j < n:
            if not stcorr_only:
                A[i, -1] = -np.log10(dist[j] / 17.)
                A[i, -2] = -(dist[j] - 17.)
            A[i, uiddic[uid]] = 1.
            tr_id = tr_ids[j]
            A[i, int(compdic[tr_id])] = -1.

            if stcorr_only:
                b[i] = np.log10(amps[j] * 1000.) + \
                    (n * np.log10(dist[j] / 17.)) + \
                    (k * (dist[j] - 17.)) + 2
            else:
                b[i] = np.log10(amps[j] * 1000.) + 2

            i += 1
            j += 1
    
    # add final constraint that the stations corrections must sum to zero
    A[i, np.linspace(0, ncomps - 1, ncomps, dtype=np.int) + nev] = 1.
    # print(A)

    A = A.tocsr()
    # result = sparse.linalg.lsqr(A, b)
    result = sparse.linalg.spsolve(A.T.dot(A), A.T.dot(b))
    
    # print('magnitudes')
    # print(result[:nev])
    # print('station corr')
    # print(result[nev:nev+ncomps])
    # print('n, k')
    # print(result[nev+ncomps:])
    # print('residuals')
    # print(_res)
    # print(result.shape)

    if not stcorr_only:
        return result[:nev], dict([(tr_id, result[int(compdic[tr_id])]) for tr_id in compdic.keys()]), result[nev+ncomps:]
    else:
        return result[:nev], result[nev:nev+ncomps]

if __name__ == "__main__":
    plt.close('all')
    obs, mags, stcorr, loc = generate_synthetic_cat(10000, 100, 1e-5, 
                                (-50, 50), (-50, 50), (0, 40), 
                                (1.3, np.nan), 0.3)

    m, s, nk = invert_mag_scale(obs)
    k, n = nk
    print((mags - m).mean(), (mags - m).std())
    # print(1.196997, n, 0.001066, k)
    # for key in sorted(s): 
    #     print('{:s} {:6.4f} {:6.4f}'.format(key, s[key], stcorr[int(key.split('.')[1])]))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.hist(mags, bins=np.linspace(0, 4, 41))
    plt.subplot(2, 2, 2)
    plt.hist(m, bins=np.linspace(-2, 6, 81))
    plt.subplot(2, 2, 3)
    plt.hist(loc[0][:, 2], bins=np.linspace(0, 40, 41))
    plt.xlabel('Depth')
    plt.subplot(2, 2, 4)
    dists = []
    for key in obs.keys():
        dists += list(obs[key]['epi_dist'].values)
    plt.hist(dists, bins=np.linspace(0, 150, 151))
    plt.xlabel('distance')
    # plt.show()

    # print(obs)

    plt.figure()
    plt.semilogy(obs[0]['epi_dist'], obs[0]['S_amp'], 'k+')
    x = np.linspace(0.1, 150, 1000)
    plt.plot(x, np.power(10., mags[0] - logA0(x, 'keir2006')) / 1000., 'k-')
    # zz = loc[0][0, 2]
    # plt.plot(x, np.power(10., mags[0] - logA0((x, zz), 'parametric', X=X, Z=Z, A=A)) / 1000., 'r-', alpha=0.5)
    # plt.plot(x, np.power(10., mags[0] - logA0((x, zz), 'parametric', X=X_out, Z=Z_out, A=attenuation)) / 1000., 'b-')
    # plt.plot(x, np.power(10., m[0] - logA0((x, zz), 'parametric', X=X_out, Z=Z_out, A=attenuation)) / 1000., 'b--')
    plt.show()
            



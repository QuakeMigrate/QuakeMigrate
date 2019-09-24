from obspy import read, UTCDateTime, read_inventory, Stream, Trace
from obspy.io.xseed import Parser
from obspy.geodetics.base import gps2dist_azimuth
from matplotlib import pyplot as plt
from obspy.signal.invsim import paz_2_amplitude_value_of_freq_resp, paz_2_amplitude_value_of_freq_resp
from scipy.signal import find_peaks
import pandas as pds
import numpy as np
import os

WOODANDERSON = {'poles' : [-5.49779 + 5.60886j, -5.49779 - 5.60886j],
                'zeros' : [0j, 0j],
                'sensitivity' : 2080,
                'gain' : 1.}

def calcMagnitude(key,a,r,method,station_corrections,verbose=True,n=False,k=False):

    try:
        corr=station_corrections[key]
    except KeyError:
        if verbose:
            print('No station corrections for',key)
        corr=0.
    return np.log10(a) + logA0(r, method, n=n, k=k) + corr

def logA0(r,method,k=False,n=False):
    if method == 'keir2006':
        return (1.196997*np.log10(r/17)) + (0.001066*(r-17)) + 2
    elif method == 'Danakil2017':
        return 1.274336*np.log10(r/17.)-0.000273*(r-17.) + 2
    elif method == 'Iceland':
        return 1.25336430829*np.log10(r/17.)+0.00315712207087*(r-17.) + 2
    elif method == 'HB':
        return 1.11*np.log10(r/100.)+0.00189*(r-100.)+3.
    elif method == 'langston1998':
        return 0.776*np.log10(r/17) + 0.000902*(r - 17) + 2.0
    elif method == 'UK':
        return 0.95*np.log10(r/100.) + 0.00183*(r-100.) -1.76 + (0.183 + 1.9 +np.log10(1./2080.)+6)
    elif method == 'Norway':
        return 1.02*np.log10(r/60.) + 0.0008*(r - 60) + 2.68
    elif n and k:
        return n*np.log10(r/17.)+k*(r-17.) + 2
    else:
        raise ValueError(method,'is not a valid method')

plt.cla()

evpath = os.path.join('out', 'test', 'events')
ppath = os.path.join('out', 'test', 'picks')
wfpath = "/raid1/tg286/ethiopia/riftvolc_data/"
inv = read_inventory(os.path.join(wfpath, 'dataless', 'dataless.seed.xml'))
dataless_file = os.path.join(wfpath, 'dataless', 'dataless.seed')
dataless = Parser(dataless_file)
events = sorted(os.listdir(evpath))
nevents = len(events)
vpvs = 1.73
noise_win = 10.
pre_pad = 10.
post_pad = 30.

prefilt = (0.1, 0.2, 20, 25)

ii = 1
while ii < nevents:
    evf = events[ii]
    print(evf)
    uid = evf.split('.')[0]
    evdata = pds.read_csv(os.path.join(evpath, evf))
    evdata['DT'] = evdata['DT'].apply(UTCDateTime)
    pdata = pds.read_csv(os.path.join(ppath, uid + '.picks'))
    pdata['ModelledTime'] = pdata['ModelledTime'].apply(UTCDateTime)
    # pdata['PickTime'].apply(UTCDateTime)

    otime = evdata['DT'].values[0]
    evla = evdata['LocalGaussian_Y'].values[0]
    evlo = evdata['LocalGaussian_X'].values[0]
    evdp = -evdata['LocalGaussian_Z'].values[0]
    wfdata = read(os.path.join(wfpath, str(otime.year), 
                    '{:03d}'.format(otime.julday), '*'))
    # print(evdata)
    # print(pdata)

    max_modelled_pick = np.max(pdata['ModelledTime'])
    wfdata.trim(otime - noise_win - pre_pad, max_modelled_pick + post_pad)
    wfdata.detrend('linear')
    wfdata.taper(type='cosine', max_percentage=0.05)
    wfdata.attach_response(inv)
    # wfdata.filter('highpass', freq=1.)

    st_wa = Stream()

    index = {}
    npicks = len(pdata)
    amplitudes = np.zeros((int(npicks * 4), 13))
    i = 0
    row = 0
    while i < npicks:
        station = pdata['Name'].values[i]
        # if not station == 'GULA':
        #     i += 2
        #     continue
        pick_1 = pdata['PickTime'].values[i]
        pick_1_phase = pdata['Phase'].values[i]
        if pick_1 == '-1':
            pick_1 = pdata['ModelledTime'].values[i]
            pick_1_phase += 'm'
        else:
            pick_1 = UTCDateTime(pick_1)

        jump = 1
        if pdata['Name'].values[i+1] == station:
            pick_2 = pdata['PickTime'].values[i + 1]
            pick_2_phase = pdata['Phase'].values[i + 1]
            jump = 2
        else:
            pick_2 = None
            pick_2_phase = None
        
        if pick_2 and pick_2 == '-1':
            pick_2 = pdata['ModelledTime'].values[i+1]
            pick_2_phase += 'm'
        elif pick_2:
            pick_2 = UTCDateTime(pick_2)
        elif not pick_2:
            if 'P' in pick_1_phase:
                pick_2 = otime + (pick_1 - otime) * vpvs
                pick_2_phase = 'Sc'
            else:
                pick_2 = otime + (pick_1 - otime) / vpvs
                pick_2_phase = 'Pc'
            
        # print(station, pick_1_phase, pick_2_phase)
        st = wfdata.copy().select(station=station)
        st_count = st.copy()
        if len(st) == 0:
            i += jump
            continue

        # st.remove_response(output='VEL', pre_filt=None, taper=False, plot=False, 
        #                 water_level=30)
        
        coords = inv.get_coordinates(st[0].get_id())
        edist, _, _ = gps2dist_azimuth(evla, evlo,
                                coords['latitude'], coords['longitude'])
        edist /= 1000.
        zdist = evdp + coords['elevation']
        zdist /= 1000.

    
        
        # st.remove_response(output='DISP', pre_filt=None, taper=False, plot=False, 
        #                 water_level=30)

        st.simulate(seedresp={'filename': dataless, 'units': "DIS"}, paz_simulate=WOODANDERSON, 
                    pre_filt=None, taper=False, water_level=30)
        # create the north-east comp combined trace
        east = st.copy().select(channel='*E')[0]
        north = st.copy().select(channel='*N')[0]
        r = np.sqrt(np.square(east) + np.square(north))
        stats = east.stats
        east.stats.channel = east.stats.channel[:-1] + "NE"
        east_north = Trace(r, east.stats)
        st += east_north
        st_wa += st.copy()
        
        # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
        for j, (tr, tr_count) in enumerate(zip(st, st_count)):
            print(tr)
            # print(tr_count)
            index[tr.id] = row
            # print(tr.id, row)
            # plt.figure()
            # plt.plot(tr.times(), tr.data, 'k-')
            
            picks = sorted([[pick_1, pick_1_phase], [pick_2, pick_2_phase]])
            windows = [[picks[0][0], picks[1][0] - 1.],
                       [picks[1][0] - 1., picks[1][0] + noise_win]]
            # plt.axvspan(picks[0][0] - tr.stats.starttime, picks[1][0] - tr.stats.starttime - 1., fc='r', zorder=-1, alpha=0.5)
            # plt.axvspan(picks[1][0] - tr.stats.starttime - 1., picks[1][0] - tr.stats.starttime + noise_win, fc='b', zorder=-1, alpha=0.5)

            # grab the maximum amplitudes
            amplitudes[row, 0:2] = edist, zdist
            k = 2
            for stime, etime in windows:
                data = tr.slice(stime, etime)
                data_count = tr_count.slice(stime, etime)
                amplitudes[row, k] += np.max(np.abs(data.data))

                # full peak2peak amplitude of WA trace
                peaks, _ = find_peaks(data.data, prominence=0.5 * amplitudes[row, k])
                troughs, _ = find_peaks(-data.data, prominence=0.5 * amplitudes[row, k])
                # print('peaks', len(peaks), 'troughs', len(troughs))

                full_amp = False
                if len(peaks) == 0 or len(troughs) == 0:
                    full_amp = -9.9

                elif len(peaks) == 1 and len(troughs) == 1:
                    full_amp = np.abs(data.data[peaks] - data.data[troughs])

                elif len(peaks) == len(troughs) and peaks[0] < troughs[0]:
                    full_peak_1 = np.abs(data.data[peaks] - data.data[troughs])
                    full_peak_2 = np.abs(data.data[peaks[1:]] - data.data[troughs[:-1]])
                    if np.max(full_peak_1) >= np.max(full_peak_2):
                        pos = np.argmax(full_peak_1)
                        full_amp = np.max(full_peak_1)
                    else:
                        pos = np.argmax(full_peak_2)
                        peaks = peaks[1:]
                        troughs = troughs[:-1]
                        full_amp = np.max(full_peak_2)

                elif len(peaks) == len(troughs) and peaks[0] > troughs[0]:
                    full_peak_1 = np.abs(data.data[peaks] - data.data[troughs])
                    full_peak_2 = np.abs(data.data[peaks[:-1]] - data.data[troughs[1:]])
                    if np.max(full_peak_1) >= np.max(full_peak_2):
                        pos = np.argmax(full_peak_1)
                        full_amp = np.max(full_peak_1)
                    else:
                        pos = np.argmax(full_peak_2)
                        peaks = peaks[:-1]
                        troughs = troughs[1:]
                        full_amp = np.max(full_peak_2)
                    
                elif not np.abs(len(peaks) - len(troughs)) == 1:
                    raise ValueError('HELP')

                elif len(peaks) > len(troughs):
                    assert peaks[0] < troughs[0]
                    full_peak_1 = np.abs(data.data[peaks[:-1]] - data.data[troughs])
                    full_peak_2 = np.abs(data.data[peaks[1:]] - data.data[troughs])
                    if np.max(full_peak_1) >= np.max(full_peak_2):
                        pos = np.argmax(full_peak_1)
                        peaks = peaks[:-1]
                        troughs = troughs
                        full_amp = np.max(full_peak_1)
                    else:
                        pos = np.argmax(full_peak_2)
                        peaks = peaks[1:]
                        troughs = troughs
                        full_amp = np.max(full_peak_2)

                elif len(peaks) < len(troughs):
                    assert peaks[0] > troughs[0]
                    full_peak_1 = np.abs(data.data[peaks] - data.data[troughs[1:]])
                    full_peak_2 = np.abs(data.data[peaks] - data.data[troughs[:-1]])
                    if np.max(full_peak_1) >= np.max(full_peak_2):
                        pos = np.argmax(full_peak_1)
                        peaks = peaks
                        troughs = troughs[1:]
                        full_amp = np.max(full_peak_1)
                    else:
                        pos = np.argmax(full_peak_2)
                        peaks = peaks
                        troughs = troughs[:-1]
                        full_amp = np.max(full_peak_2)
                
                k += 3
                amplitudes[row, k] = full_amp
                # print(full_amp)
                
                
                peaks, _ = find_peaks(data_count.data, prominence=0.5 * amplitudes[row, k])
                troughs, _ = find_peaks(-data_count.data, prominence=0.5 * amplitudes[row, k])
                # print('peaks', len(peaks), 'troughs', len(troughs))

                full_amp = False
                if len(peaks) == 0 or len(troughs) == 0:
                    k += 3
                    amplitudes[row, k] = -9.9
                    k += 2 
                    amplitudes[row, k] = -9.9
                    k -= 7
                    continue

                elif len(peaks) == 1 and len(troughs) == 1:
                    full_amp = np.abs(data_count.data[peaks] - data_count.data[troughs])

                elif len(peaks) == len(troughs) and peaks[0] < troughs[0]:
                    full_peak_1 = np.abs(data_count.data[peaks] - data_count.data[troughs])
                    full_peak_2 = np.abs(data_count.data[peaks[1:]] - data_count.data[troughs[:-1]])
                    if np.max(full_peak_1) >= np.max(full_peak_2):
                        pos = np.argmax(full_peak_1)
                        full_amp = np.max(full_peak_1)
                    else:
                        pos = np.argmax(full_peak_2)
                        peaks = peaks[1:]
                        troughs = troughs[:-1]
                        full_amp = np.max(full_peak_2)

                elif len(peaks) == len(troughs) and peaks[0] > troughs[0]:
                    full_peak_1 = np.abs(data_count.data[peaks] - data_count.data[troughs])
                    full_peak_2 = np.abs(data_count.data[peaks[:-1]] - data_count.data[troughs[1:]])
                    if np.max(full_peak_1) >= np.max(full_peak_2):
                        pos = np.argmax(full_peak_1)
                        full_amp = np.max(full_peak_1)
                    else:
                        pos = np.argmax(full_peak_2)
                        peaks = peaks[:-1]
                        troughs = troughs[1:]
                        full_amp = np.max(full_peak_2)
                    
                elif not np.abs(len(peaks) - len(troughs)) == 1:
                    raise ValueError('HELP')

                elif len(peaks) > len(troughs):
                    assert peaks[0] < troughs[0]
                    full_peak_1 = np.abs(data_count.data[peaks[:-1]] - data_count.data[troughs])
                    full_peak_2 = np.abs(data_count.data[peaks[1:]] - data_count.data[troughs])
                    if np.max(full_peak_1) >= np.max(full_peak_2):
                        pos = np.argmax(full_peak_1)
                        peaks = peaks[:-1]
                        troughs = troughs
                        full_amp = np.max(full_peak_1)
                    else:
                        pos = np.argmax(full_peak_2)
                        peaks = peaks[1:]
                        troughs = troughs
                        full_amp = np.max(full_peak_2)

                elif len(peaks) < len(troughs):
                    assert peaks[0] > troughs[0]
                    full_peak_1 = np.abs(data_count.data[peaks] - data_count.data[troughs[1:]])
                    full_peak_2 = np.abs(data_count.data[peaks] - data_count.data[troughs[:-1]])
                    if np.max(full_peak_1) >= np.max(full_peak_2):
                        pos = np.argmax(full_peak_1)
                        peaks = peaks
                        troughs = troughs[1:]
                        full_amp = np.max(full_peak_1)
                    else:
                        pos = np.argmax(full_peak_2)
                        peaks = peaks
                        troughs = troughs[:-1]
                        full_amp = np.max(full_peak_2)

                peak_t = data_count.times()[peaks[pos]]
                trough_t = data_count.times()[troughs[pos]]
                approx_freq = 1. / (np.abs(peak_t - trough_t) * 2.)
                
                gain = paz_2_amplitude_value_of_freq_resp(WOODANDERSON, approx_freq) * WOODANDERSON['sensitivity']
                gain /= np.abs(tr_count.stats.response.get_evalresp_response_for_frequencies([approx_freq], output='DISP'))
                k += 3
                amplitudes[row, k] = full_amp * gain
                # print(full_amp, gain, approx_freq, full_amp * gain)
                k += 2 
                amplitudes[row, k] = approx_freq
                k -= 7
            # plt.axhline(amplitudes[row, 2], c='r')
            # plt.axhline(amplitudes[row, 3], c='b')
            # plt.axhline(-amplitudes[row, 2], c='r')
            # plt.axhline(-amplitudes[row, 3], c='b')

            # and calculate a noise
            data = tr.slice(pick_1 - 3. - noise_win, pick_1 - 3.)
            amplitudes[row, 4] = np.std(data.data)
            amplitudes[row, 7] = np.std(data.data) * 2
            amplitudes[row, 12] = np.std(data.data) * 2

            # plt.axhline(amplitudes[row, 4], c='gray')
            # plt.axhline(4 * amplitudes[row, 4], c='gray')
            # plt.axhline(-amplitudes[row, 4], c='gray')
            # plt.axhline(-4 * amplitudes[row, 4], c='gray')
            # plt.title(tr.id)
            # plt.show()
            # plt.savefig(tr.id + '.pdf')
            # plt.cla()
            row += 1
        i += jump
    # sys.exit()
    

    amplitudes = pds.DataFrame(amplitudes[:row, :], 
        columns=['epi_dist', 'depth', 
                'wa_P_amp', 'wa_S_amp', 'wa_Error', 
                'wa_P_amp_full', 'wa_S_amp_full', 'wa_Error_full',
                'count_P_amp_full', 'count_S_amp_full', 'count_P_freq', 'count_S_freq', 'count_Error_full'],
        index=index)
    amplitudes.to_csv(otime.strftime('%Y%m%d%H%M%S.csv'))
    amp_ne = amplitudes.filter(regex='.[BH]HNE$', axis=0)
    amp_z = amplitudes.filter(regex='.[BH]HZ$', axis=0)
    amp_n = amplitudes.filter(regex='.[BH]HN$', axis=0)
    amp_e = amplitudes.filter(regex='.[BH]HE$', axis=0)

    # filter by error
    amp_e_efilt = amp_e[(amp_e['wa_S_amp'] > 4. * amp_e['wa_Error'])]
    amp_n_efilt = amp_n[(amp_n['wa_S_amp'] > 4. * amp_n['wa_Error'])]
    amp_z_efilt = amp_z[amp_z['wa_S_amp'] > 4. * amp_z['wa_Error']]
    amp_ne_efilt = amp_ne[amp_ne['wa_S_amp'] > 4. * amp_ne['wa_Error']]

    plt.figure(10)
    plt.semilogy(amp_e['epi_dist'], 1000. * amp_e['wa_S_amp'], 'ko', label='WA')
    plt.semilogy(amp_e['epi_dist'], 1000. * amp_e['wa_S_amp_full'] / 2., 'ro', label='WA_full')
    plt.semilogy(amp_e['epi_dist'], 1000. * amp_e['count_S_amp_full'] / 2., 'bo', label='count_full')
    plt.semilogy(amp_n['epi_dist'], 1000. * amp_n['wa_S_amp'], 'k+')
    plt.semilogy(amp_n['epi_dist'], 1000. * amp_n['wa_S_amp_full'] / 2., 'r+')
    plt.semilogy(amp_n['epi_dist'], 1000. * amp_n['count_S_amp_full'] / 2., 'b+')
    x = np.linspace(0, np.ceil(np.max(amp_n['epi_dist'])))
    for m in [1., 2., 3., 4.]:
        plt.plot(x , np.power(10., m - logA0(x, 'keir2006')), 'k--')
        plt.text(x[0], np.power(10., m - logA0(x, 'keir2006'))[0], 'M = ' + str(m))
    plt.legend(loc=0)

    # plt.figure(10)
    # plt.gca().set_yscale('log') 
    # plt.errorbar(amp_ne_efilt['epi_dist'], amp_ne_efilt['wa_S_amp'] * 1000., 
    #             yerr=amp_ne_efilt['wa_Error'], c='r', marker='o', ls='none')
    # plt.errorbar(amp_ne['epi_dist'], amp_ne['wa_S_amp'] * 1000., 
    #             yerr=amp_ne['wa_Error'], c='r', marker='o', mfc='none', ls='none')
    
    # for m in [1., 2., 3., 4.]:
    #     plt.plot(x , np.power(10., m - logA0(x, 'keir2006')), 'k--')
    #     plt.text(x[0], np.power(10., m - logA0(x, 'keir2006'))[0], 'M = ' + str(m))

    
    plt.title('Wood-Anderson')
    plt.xlabel('Distance / km')
    plt.ylabel('\$amplitude [m]\$')

    mags_wa_e = calcMagnitude('', amp_e['wa_S_amp'].values * 1000., amp_e['epi_dist'].values, 'keir2006', {})
    mags_wa_full_e = calcMagnitude('', amp_e['wa_S_amp_full'].values * 1000., amp_e['epi_dist'].values, 'keir2006', {})
    mags_count_full_e = calcMagnitude('', amp_e['count_S_amp_full'].values * 1000., amp_e['epi_dist'].values, 'keir2006', {})
    mags_wa_n = calcMagnitude('', amp_n['wa_S_amp'].values * 1000., amp_n['epi_dist'].values, 'keir2006', {})
    mags_wa_full_n = calcMagnitude('', amp_n['wa_S_amp_full'].values * 1000., amp_n['epi_dist'].values, 'keir2006', {})
    mags_count_full_n = calcMagnitude('', amp_n['count_S_amp_full'].values * 1000., amp_n['epi_dist'].values, 'keir2006', {})



    mag_wa = (np.mean(mags_wa_e) + np.mean(mags_wa_n)) / 2.
    mag_wa_full = (np.mean(mags_wa_full_e) + np.mean(mags_wa_full_n)) / 2.
    mag_count_full = (np.mean(mags_count_full_e) + np.mean(mags_count_full_n)) / 2.
    print(mag_wa, mag_wa_full, mag_count_full)

    # plt.figure()
    # plt.hist(mags_ne, np.linspace(1, 3.5, 26), alpha=0.5)
    # plt.axvline(mean_mag)
    # plt.savefig(otime.strftime('%Y%m%d%H%M%S_maghist.pdf'))

    plt.figure(10)
    for m in [mag_wa, mag_wa_full, mag_count_full]:
        plt.plot(x , np.power(10., m - logA0(x, 'keir2006')), 'k-', lw=2)
    # plt.savefig(otime.strftime('%Y%m%d%H%M%S_magnitudes.pdf'))
    plt.show()

    # for station in set([tr.stats.station for tr in st_wa]):
    #     ohit = st_wa.select(station=station, channel='?HNE')[0]
    #     plt.figure()
    #     plt.plot(ohit.times(), ohit.data, 'k-')

    #     pick = pdata[pdata['Name'] == station]
    #     P = pick[pick['Phase'] == 'P']['ModelledTime'].values[0]
    #     plt.axvline(P - ohit.stats.starttime, c='r')
    #     S = pick[pick['Phase'] == 'S']['ModelledTime'].values[0]
    #     plt.axvline(S - ohit.stats.starttime, c='b')
    #     plt.axvline(S - ohit.stats.starttime + noise_win, c='b')

    #     a = amp_ne.filter(regex=station, axis=0)
    #     print(a)
    #     for i in [1]: #, -1]:
    #         plt.axhline(i * a['wa_S_amp'].values[0], c='b')
    #         plt.axhline(i * a['wa_P_amp'].values[0], c='r')
    #         plt.axhline(i * a['wa_Error'].values[0])
    #         # plt.axhline(i * a['wa_Error'].values[0] * 2)
    #         # plt.axhline(i * a['wa_Error'].values[0] * 3)
    #         plt.axhline(i * a['wa_Error'].values[0] * 4)
    #         # plt.axhline(i * a['wa_Error'].values[0] * 5)
    #         # plt.axhline(i * a['wa_Error'].values[0] * 6)
    #     plt.title(station + str(a['epi_dist'].values[0]))
    #     plt.show()
    plt.cla()
    ii += 1
    break
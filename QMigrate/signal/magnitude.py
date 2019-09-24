from obspy import read, UTCDateTime, read_inventory, Stream
from obspy.geodetics.base import gps2dist_azimuth
from matplotlib import pyplot as plt
from obspy.signal.invsim import WOODANDERSON
import pandas as pds
import numpy as np
import os

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
events = sorted(os.listdir(evpath))
nevents = len(events)
vpvs = 1.73
noise_win = 10.
pre_pad = 10.
post_pad = 30.

prefilt = (0.1, 0.2, 20, 25)

ii = 0
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
    wfdata.filter('highpass', freq=1.)

    st_wa = Stream()

    index = {}
    npicks = len(pdata)
    amplitudes = np.zeros((int(npicks * 3), 11))
    i = 0
    row = 0
    while i < npicks:
        station = pdata['Name'].values[i]
        # if not station == 'OHIT':
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
            
        print(station, pick_1_phase, pick_2_phase)
        st = wfdata.copy().select(station=station)
        if len(st) == 0:
            i += jump
            continue

        st.remove_response(output='VEL', pre_filt=None, taper=False, plot=False, 
                        water_level=30)
        
        coords = inv.get_coordinates(st[0].get_id())
        edist, _, _ = gps2dist_azimuth(evla, evlo,
                                coords['latitude'], coords['longitude'])
        edist /= 1000.
        zdist = evdp + coords['elevation']
        zdist /= 1000.
        # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
        for j, tr in enumerate(st):
            print(tr.id, row)
            # ax = axes[j]
            # ax.plot(tr.times(), tr.data, 'k-')
            # ax.axvline(otime - tr.stats.starttime, c='k')

            # if 'm' in pick_1_phase:
            #     ax.axvline(pick_1 - tr.stats.starttime, c='r', ls='--')
            # else:
            #     ax.axvline(pick_1 - tr.stats.starttime, c='r')
            
            # if pick_2_phase and 'm' in pick_2_phase:
            #     ax.axvline(pick_2 - tr.stats.starttime, c='b', ls='--')
            # elif pick_2_phase:
            #     ax.axvline(pick_2 - tr.stats.starttime, c='b')
            
            index[tr.id] = row
            amplitudes[row, 0:2] = edist, zdist
            picks = sorted([[pick_1, pick_1_phase], [pick_2, pick_2_phase]])
            windows = [[picks[0][0], picks[1][0]],
                       [picks[1][0], picks[1][0] + noise_win]]

            # grab the maximum amplitudes
            k = 2
            for stime, etime in windows:
                data = tr.slice(stime, etime)
                amplitudes[row, k] += np.max(np.abs(data.data))
                k += 1
            # and calculate a noise
            data = tr.slice(pick_1 - 3. - noise_win, pick_1 - 3.)
            amplitudes[row, k] = np.std(data.data)
            k += 1
            row += 1
        
        st = wfdata.copy().select(station=station)
        st.remove_response(output='DISP', pre_filt=None, taper=False, plot=False, 
                        water_level=30)
        
        # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
        for j, tr in enumerate(st):
            row = index[tr.id]
            print(tr.id, row)
            # ax = axes[j]
            # ax.plot(tr.times(), tr.data, 'k-')
            # ax.axvline(otime - tr.stats.starttime, c='k')

            # if 'm' in pick_1_phase:
            #     ax.axvline(pick_1 - tr.stats.starttime, c='r', ls='--')
            # else:
            #     ax.axvline(pick_1 - tr.stats.starttime, c='r')
            
            # if pick_2_phase and 'm' in pick_2_phase:
            #     ax.axvline(pick_2 - tr.stats.starttime, c='b', ls='--')
            # elif pick_2_phase:
            #     ax.axvline(pick_2 - tr.stats.starttime, c='b')
            
            picks = sorted([[pick_1, pick_1_phase], [pick_2, pick_2_phase]])
            windows = [[picks[0][0], picks[1][0]],
                       [picks[1][0], picks[1][0] + noise_win]]

            # grab the maximum amplitudes
            k = 5
            for stime, etime in windows:
                data = tr.slice(stime, etime)
                amplitudes[row, k] += np.max(np.abs(data.data))
                k += 1
            # and calculate a noise
            data = tr.slice(pick_1 - 3. - noise_win, pick_1 - 3.)
            amplitudes[row, k] = np.std(data.data)
            k += 1

        st.simulate(paz_simulate=WOODANDERSON)
        st_wa += st.copy()
        # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
        for j, tr in enumerate(st):
            row = index[tr.id]
            print(tr.id, row)
            # ax = axes[j]
            # ax.plot(tr.times(), tr.data, 'k-')
            # ax.axvline(otime - tr.stats.starttime, c='k')

            # if 'm' in pick_1_phase:
            #     ax.axvline(pick_1 - tr.stats.starttime, c='r', ls='--')
            # else:
            #     ax.axvline(pick_1 - tr.stats.starttime, c='r')
            
            # if pick_2_phase and 'm' in pick_2_phase:
            #     ax.axvline(pick_2 - tr.stats.starttime, c='b', ls='--')
            # elif pick_2_phase:
            #     ax.axvline(pick_2 - tr.stats.starttime, c='b')
            
            picks = sorted([[pick_1, pick_1_phase], [pick_2, pick_2_phase]])
            windows = [[picks[0][0], picks[1][0]],
                       [picks[1][0], picks[1][0] + noise_win]]

            # grab the maximum amplitudes
            k = 8
            for stime, etime in windows:
                data = tr.slice(stime, etime)
                amplitudes[row, k] += np.max(np.abs(data.data))
                k += 1
            # and calculate a noise
            data = tr.slice(pick_1 - 3. - noise_win, pick_1 - 3.)
            amplitudes[row, k] = np.std(data.data)

        row += 1
        i += jump
        

    break

amplitudes = pds.DataFrame(amplitudes[:row, :], 
    columns=['epi_dist', 'depth', 
            'vel_P_amp', 'vel_S_amp', 'vel_Error',
            'disp_P_amp', 'disp_S_amp', 'disp_Error',
            'wa_P_amp', 'wa_S_amp', 'wa_Error'],
    index=index)
amp_ne = amplitudes.filter(regex='.[BH]H[NE]$', axis=0)
amp_z = amplitudes.filter(regex='.[BH]HZ$', axis=0)
amp_n = amplitudes.filter(regex='.[BH]HN$', axis=0)
amp_e = amplitudes.filter(regex='.[BH]HE$', axis=0)

# filter by error
amp_e_efilt = amp_e[(amp_e['wa_S_amp'] > 4. * amp_e['wa_Error'])]
amp_n_efilt = amp_n[(amp_n['wa_S_amp'] > 4. * amp_n['wa_Error'])]
amp_z_efilt = amp_z[amp_z['wa_S_amp'] > 4. * amp_z['wa_Error']]
amp_ne_efilt = amp_ne[amp_ne['wa_S_amp'] > 4. * amp_ne['wa_Error']]

ne_comb = np.sqrt(np.square(amp_n['wa_S_amp'].values) + np.square(amp_e['wa_S_amp'].values))
ne_comb_err = np.sqrt((np.square(amp_n['wa_S_amp'].values * amp_n['wa_Error'].values) + \
                np.square(amp_e['wa_S_amp'].values * amp_e['wa_Error'].values)) / \
                (np.square(amp_n['wa_S_amp'].values) + np.square(amp_e['wa_S_amp'].values)))

ne_comb_efilt = ne_comb[ne_comb > 4. * ne_comb_err]
ne_comb_efilt_err = ne_comb_err[ne_comb > 4. * ne_comb_err]
ne_comb_efilt_edist = amp_n['epi_dist'][ne_comb > 4. * ne_comb_err]

# plt.figure()
# plt.errorbar(amplitudes['epi_dist'], amplitudes['vel_P_amp'], 
#             yerr=amplitudes['vel_Error'], c='b', marker='o', ls='none')
# plt.errorbar(amplitudes['epi_dist'], amplitudes['vel_S_amp'], 
#             yerr=amplitudes['vel_Error'], c='r', marker='o', ls='none')
# plt.gca().set_yscale('log') 
# plt.title('velocity')

# plt.figure()
# plt.errorbar(amplitudes['epi_dist'], amplitudes['disp_P_amp'], 
#             yerr=amplitudes['disp_Error'], c='b', marker='o', ls='none')
# plt.errorbar(amplitudes['epi_dist'], amplitudes['disp_S_amp'], 
#             yerr=amplitudes['disp_Error'], c='r', marker='o', ls='none')
# plt.gca().set_yscale('log') 
# plt.title('displacement')

plt.figure(10)
# plt.errorbar(amplitudes['epi_dist'], np.log10(amplitudes['wa_P_amp'] * 1000.), 
            # yerr=amplitudes['wa_Error'], c='b', marker='o', ls='none')
# plt.errorbar(amp_ne['epi_dist'], np.log10(amp_ne['wa_S_amp'] * 1000.), 
#             yerr=amp_ne['wa_Error'], c='r', marker='o', ls='none')
plt.errorbar(ne_comb_efilt_edist, np.log10(ne_comb_efilt * 1000.), 
            yerr=ne_comb_efilt_err, c='r', marker='o', ls='none')
plt.errorbar(amp_n['epi_dist'], np.log10(ne_comb * 1000.), 
            yerr=ne_comb_err, mfc='none', marker='o', ls='none')
for m in [1., 2., 3., 4.]:
    x = np.sort(amplitudes['epi_dist'])
    plt.plot(x , m - logA0(x, 'keir2006'), 'k--')

# plt.gca().set_yscale('log') 
plt.title('Wood-Anderson')
plt.xlabel('Distance / km')
plt.ylabel('\$log10(amplitude [mm])\$')

plt.figure()
mags_ne = calcMagnitude('', amp_ne['wa_S_amp'] * 1000., amp_ne['epi_dist'], 'keir2006', {})
mags_z = calcMagnitude('', amp_z['wa_S_amp'] * 1000., amp_z['epi_dist'], 'keir2006', {})

mags_ne_comb = calcMagnitude('', ne_comb * 1000., amp_z['epi_dist'], 'keir2006', {})

plt.plot(amp_ne['epi_dist'], mags_ne, 'ko')
plt.plot(amp_z['epi_dist'], mags_z, 'ro')
plt.plot(amp_z['epi_dist'], mags_ne_comb, 'bo')

plt.figure()
plt.hist(mags_ne, np.linspace(1, 3.5, 26), alpha=0.5)
plt.hist(mags_z, np.linspace(1, 3.5, 26), alpha=0.5)
plt.hist(mags_ne_comb, np.linspace(1, 3.5, 26), alpha=0.5)

mean_mag = np.mean(mags_ne)

plt.figure(10)
x = np.sort(amplitudes['epi_dist'])
plt.plot(x , mean_mag - logA0(x, 'keir2006'), 'k-', lw=2)
plt.plot(x , np.mean(mags_ne_comb) - logA0(x, 'keir2006'), 'r-', lw=2)
plt.show()

for station in set([tr.stats.station for tr in st_wa]):
    ohit = st_wa.select(station=station, channel='?HE')[0]
    plt.figure()
    plt.plot(ohit.times(), ohit.data, 'k-')
    a = amp_e.filter(regex=station, axis=0)
    for i in [1, -1]:
        plt.axhline(i * a['wa_S_amp'].values[0], c='b')
        plt.axhline(i * a['wa_P_amp'].values[0], c='r')
        plt.axhline(i * a['wa_Error'].values[0])
        plt.axhline(i * a['wa_Error'].values[0] * 2)
        plt.axhline(i * a['wa_Error'].values[0] * 3)
        plt.axhline(i * a['wa_Error'].values[0] * 4)
        plt.axhline(i * a['wa_Error'].values[0] * 5)
        plt.axhline(i * a['wa_Error'].values[0] * 6)
    plt.title(station + str(a['epi_dist'].values[0]))
    plt.show()
# this script uses the quakemigrate output format and calculates moment magnitude
# using the method by Abercrombie (REFERENCE) and Greenfield et al. (REFERENCE)
# broadly, QM dervived pick times and waveforms are selected and the instrument response
# is removed. Using the pick times (either modelled or picked) windows are selected
# around the P and S phases. A noise window is also selected from before the P arrival.
# Spectra are calculated and fit using either the brune (1980) or Boatwright (REFERENCE)
# models. These functions have three unknown variables: long-period spectral level (sigma0), 
# corner frequency (fc) and attenuation (Q). Note Fc and Q trade off extremely and can 
# rarely be trusted. After this some standard calculations are performed to calculate 
# stress drop, rupture area, moment and moment magnitude. These are output in csv
# files for each channel-station-earthquake. New columns are appended to the event
# file containing the mean and std error for each event. Some QC parameters are 
# available, but the user should consider checking the results for themselves.
import os
from argparse import ArgumentParser
from quakemigrate.io.event import Event
from quakemigrate.io.data import WaveformData
from obspy import read, read_inventory, UTCDateTime as UTC
from obspy.geodetics import gps2dist_azimuth
from mtspec import mtspec
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import pandas as pds
import numpy as np

def argument_parser():
    psr = ArgumentParser()
    psr.add_argument("qm_run", help="The QM run you want to run the script on")
    psr.add_argument("inventory", help="Inventory object containing response")
    psr.add_argument("--all_events", default=False, action="store_true",
            help="Calculate moment magnitude on all the events in the QM catalogue")
    psr.add_argument("--filelist", 
            help="Calculate moment magnitudes on only the events in the file")
    psr.add_argument("--marginal_window", default=1.)
    psr.add_argument("--prefilt", default=None, help="prefilter used when removing instrument response")
    psr.add_argument("--water_level", default=60., help="water level used removing instrument response")
    psr.add_argument("--remove_full_response", action="store_true", default=False, help="remove the full instrument response (including FIR filters)")
    psr.add_argument("--prepick", default=0.2, help="start of signal window before pick")
    psr.add_argument("--postpick", default=5., help="end of signal window after pick")
    psr.add_argument("--nfft", default=2**10, help="Length of the FFT")
    psr.add_argument("--use_model_time", action="store_true", default=False, help="Use model time picks rather than picktimes")
    psr.add_argument("--rupture_velocity", default=0.9, help="Rupture velocity as a fraction")
    psr.add_argument("--beta", default=2400, help="Shear wave velocity")


    return psr

def moment(sigma0, distance, beta=2400., rho=2700., phase='S'):
    """
    Function to calculate the moment from the long period spectral level
    beta : shear wave velocity
    rho : density
    U : average amplitude across focal sphere
    """
    
    if phase == 'S':
        U = .59 
    elif phase == 'P':
        U = .42
        beta *= np.sqrt(3)
    
    return (4.*np.pi*rho*beta**3*distance*sigma0) / (2.*U)

def radius(fc, beta=1700., phase='S', model='kaneko', Vr=0.9):
    """
    Function to fault raduis using the corner frequency
    """
    
    option = {0.9:{'kaneko':(.38, .26), 'madriaga':(.32, .21)},
              0.7:{'kaneko':(.32, .26)},
              0.5:{'kaneko':(.28, .22)},
              0.1:{'kaneko':(np.nan, .1)}}
    if phase == 'S':
        k = option[Vr][model][1]
    elif phase == 'P':
        k = option[Vr][model][0]
        beta *= np.sqrt(3)
    
    return (k*beta)/fc

def stress_drop(m0, r):
    return (7*m0) / (16.*r**3)

def moment_magnitude(moment_nm, eqn='kanamori'):
    if eqn == 'kanamori':
        return (np.log10(moment_nm)/1.5) - 6.03
    elif eqn == 'small':
        return (np.log10(moment_nm)) - 10.545
    else:
        raise AttributeError('option',eqn,'is not allowed')

def brune(f, fc, sigma0, n, gamma):
    return sigma0 / np.power((1. + np.power((f/fc),(gamma*n))),(1./gamma))

def attenuation(f, traveltime, Q):
    return np.exp((-np.pi*f*traveltime)/Q)

def geometric_spreading(d):
    return np.power(d, -1)

def site_amplification():
    return 1.

def fc_range(ML, dstress_min=0.001e6, dstress_max=100e6, beta=2400):
    
    def calc_fc(N, mag, stress, svel):
        return (1.1**N) * (0.4906*svel*(stress/(10.**((1.5*mag)+9.1)))**(1./3.))
    n = 1
    fc = calc_fc(n, ML, dstress_min, beta)
    fcs = [fc]
    n = 2
    while fc < calc_fc(0, ML, dstress_max, beta):
        fc = calc_fc(n, ML, dstress_min, beta)
        fcs.append(fc)
        n += 1
    return np.asarray(fcs)

def model(f, fc, sigma0, Q=None, traveltime=None, n=2, gamma=1, 
                inc_attenuation=False, inc_site_amplification=False, 
                inc_geometric_spreading=False, distance=None):
    """
    Brune mode as implemented by Abercrombie 1995
    n = 2, gamma=1 is the Brune (1970) model
    n=2, gamma=2 is the Boatwright (1980) model
    """
    
    if inc_attenuation and (not Q or not traveltime):
        raise AttributeError('If you need the attenuation term then you need the traveltime an Q terms')
    if inc_geometric_spreading and not distance:
        raise AttributeError('If you need the geometrical spreading term then you need the distance')
    
    if inc_attenuation:
        att = attenuation(f, traveltime, Q)
    else:
        att = 1
    if inc_geometric_spreading:
        geo_spread = geometric_spreading(distance)
    else:
        geo_spread = 1.
    if inc_site_amplification:
        site_amp = site_amplification()
    else:
        site_amp = 1.
    
    source = brune(f, fc, sigma0, n, gamma)
    
    return source*att*geo_spread*site_amp

def costFunction(f, y, corner_freq, inputs, 
                traveltime, d, gamma=1, 
                inc_attenuation=True, inc_site_amplification=False, 
                inc_geometric_spreading=True):
    sigma0 = inputs[0]
    Q = inputs[1]
    
    h = model(f, corner_freq, sigma0, gamma=gamma,
              Q=Q, traveltime=traveltime, inc_attenuation=inc_attenuation,
              inc_geometric_spreading=inc_geometric_spreading, distance=d,
              inc_site_amplification=inc_site_amplification)
    
    if np.any(h<=0):
        return 1e10
    h = np.log10(h)
    
    
    m = len(f)
    J = (1./(2*m)) * np.sum( np.power((h - y),2.))
    
    # gradient = (1./m) * np.sum( (h - y) * f)
    
    return J #, gradient

def resample_and_smooth(x_new, x, y, win_len=5, do_smoothing=True):
    # log sample the spectra
    resample = np.interp(x_new, x, y)
    
    if do_smoothing:
        # 5 point median smoothing window for the amplitude
        smooth = np.zeros_like(resample)
        n = len(resample)
        window = int(np.floor(win_len/2))
        i = 0
        while i < n:
            before = i-window
            while before < 0:
                before += 1
            
            after = i+1+window
            while after >= n:
                after -= 1
            
            # smooth[i] = np.mean(resample[before:after])
            smooth[i] = np.median(resample[before:after])
            i += 1
        return smooth
    else:
        return resample

def calculate_moment_magnitude(ev, ml, pickwindow, nfft, beta,
                        use_model_time=False, Vr=0.9):
    st = ev.data.raw_waveforms
    pkdf = ev.picks["df"]
    npick = len(ev.picks["df"])
    nchannel = len(st)

    prepick, postpick = pickwindow
    noise_window_length = postpick - prepick

    info = None
    # # calculate the spectra
    moment_mags = np.zeros(nchannel) * np.nan
    moment_0 = np.zeros_like(moment_mags) * np.nan
    stress_drops = np.zeros((nchannel,3)) * np.nan
    radii = np.zeros_like(stress_drops) * np.nan
    corner_frequencies = np.zeros_like(stress_drops) * np.nan
    Q_param = np.zeros_like(moment_0) * np.nan
    epidists = np.zeros_like(moment_mags) * np.nan
    hypdists = np.zeros_like(moment_mags) * np.nan
    phases = []
    for i in range(nchannel):
        raw = st[i]
        real = ev.data.get_real_waveform(raw.copy(), velocity=False)

        # extract some useful information
        station = real.stats.station
        if real.stats.channel[-1] == "Z":
            phase = "P"
        elif real.stats.channel[-1] in "NE":
            phase = "S"
        else:
            raise ValueError(f"Unknown channel identifier {real.stats.channel[-1]}")
        phases.append(phase)
        stcoords = ev.data.response_inv.get_coordinates(raw.id)
        ev_coords = ev.locations["spline"]
        epi_dist, _az, _baz = gps2dist_azimuth(ev_coords["Y"], ev_coords["X"],
                                                stcoords["latitude"], stcoords["longitude"])
        ddepth = (ev_coords["Z"] - (stcoords["elevation"]/1e3)) * 1e3
        hyp_dist = np.sqrt(epi_dist**2 + ddepth**2)
        epidists[i] = epi_dist
        hypdists[i] = hyp_dist
        
        # window around the pick time
        pick = pkdf[(pkdf.Station==station) & (pkdf.Phase==phase)]
        if not len(pick) == 1:
            raise ValueError(f"number of matched picks should be 1, you have {len(pick)}")
        if use_model_time:
            pick = pick.ModelledTime.iloc[0]
        else:
            pick = pick.PickTime.iloc[0]
            if pick == "-1":
                continue
            else:
                pick = UTC(pick)
        real_signal = real.copy().trim(pick-prepick, pick+postpick)
        
        # calculate the FFT and smooth
        amp_signal, freq_signal = mtspec(real_signal.data, real_signal.stats.delta, 3, nfft=nfft)
        amp_signal = np.sqrt(amp_signal)
        freq_log = np.logspace(-2, np.log10(freq_signal[-1]), 200)
        amp_signal_smooth = resample_and_smooth(freq_log, freq_signal, amp_signal, win_len=5)

        # make a noise window
        real_noise = real.copy().trim(ev.otime-noise_window_length, ev.otime)
        real_noise.detrend('linear')
        amp_noise, freq_noise = mtspec(real_noise.data, real_noise.stats.delta, 3, nfft=nfft)
        amp_noise = np.sqrt(amp_noise)
        amp_noise_smooth = resample_and_smooth(freq_log, freq_noise, amp_noise, win_len=5)
        
        # plt.figure(1)
        # plt.loglog(freq_signal, amp_signal, '-', c='gray', label='Signal')
        # plt.loglog(freq_log, amp_signal_smooth, 'k-', label='Smooth Signal')
        # plt.loglog(freq_log, amp_noise_smooth, ':', c='gray', label='Noise')
        
        # calculate the signal-to-noise ratio and do a quick QC
        SNR = amp_signal_smooth / amp_noise_smooth
        # plt.figure(2)
        # plt.semilogx(freq_log, SNR, 'k-')
        # plt.axhline(3, ls='--')
        # plt.xlabel('Frequency / Hz')
        # plt.ylabel('signal-to-noise ratio')
        if np.all(SNR<3):
            continue
        elif np.log10(max(freq_log[SNR>3])) - np.log10(min(freq_log[SNR>3])) < .5:
            continue
        elif len(freq_log[SNR>3]) < 50:
            continue
        
        
        # minmize the cost function
        # plt.figure(1)
        INITIAL_VAL = [1.e-6, 10.]
        traveltime = pick - ev.otime

        #cost = lambda inputs: costFunction(freq_log[SNR>3], np.log10(amp_signal_smooth[SNR>3]), inputs, traveltime, 0, gamma=1)
        
        # can calculate a range of corner frequencies using the magnitude
        # or just search through a logspace, remember that we loop over all
        # the corner frequencies so try to avoid making this too long
    #    fc_list = fc_range(mag)
        fc_list = np.logspace(-1, 2.5, 50)
        final_costs = np.zeros_like(fc_list)
        n_fc = len(fc_list)
        out = np.zeros((n_fc, 2)) # SIGMA COL 0, Q COL 1

        # now loop over the list of fc inverting for Q and M0 at each value
        for k, fc in enumerate(fc_list):
            cost = lambda inputs: costFunction(freq_log[SNR>3], # only fit where the SNR is above 3
                                               np.log10(amp_signal_smooth[SNR>3]), 
                                               fc, inputs, traveltime, hyp_dist/1000, gamma=1,
                                               inc_attenuation=True, inc_site_amplification=False,
                                               inc_geometric_spreading=False)
            res = minimize(cost, INITIAL_VAL, method='Nelder-Mead', jac=False)
            
            final_cost = res.fun
            final_costs[k] = final_cost
            
            sigma0 = res.x[0]
            att = res.x[1]
            out[k, 0] = sigma0
            out[k, 1] = att
        #
        min_loc = np.argmin(final_costs)
        best_fc = fc_list[min_loc]
        
        # fit the cost function to a quadratic funcion to 
        # get the best fitting corner frequency
        min_cost = np.min(final_costs)
        # fit the minima at 30% level
        fitting_area = final_costs < min_cost*1.3
        jj = 1
        while jj < len(fitting_area):
            if fitting_area[jj]:
                fitting_area[jj-1] = True
                break
            jj += 1
        tmp = fc_list[fitting_area]
        new_freq_samples = np.logspace(np.log10(np.min(tmp)), 
                                       np.log10(np.max(tmp)),
                                       100)
        p = np.polyfit(np.log10(tmp),final_costs[fitting_area], 8)
        
        new_cost_y = np.polyval(p, np.log10(new_freq_samples))
        min_cost = np.min(new_cost_y)
        
        tmp = new_freq_samples[new_cost_y < min_cost*1.1] # 10% error?
        new_best_fc = new_freq_samples[np.argmin(new_cost_y)]
        if abs(best_fc - new_best_fc) > 1:
            best_fc = best_fc
        else:
            best_fc = new_best_fc
        fc_error = [best_fc-np.min(tmp), np.max(tmp)-best_fc]
        
        
        # recalculate Q and sigma0 for the new fc
        cost = lambda inputs: costFunction(freq_log[SNR>3], 
                                           np.log10(amp_signal_smooth[SNR>3]), 
                                           best_fc, inputs, traveltime, hyp_dist/1000, gamma=1,
                                           inc_attenuation=True, inc_site_amplification=False,
                                           inc_geometric_spreading=False)
        res = minimize(cost, INITIAL_VAL, method='Nelder-Mead', jac=False)
            
        best_sigma0 = res.x[0]
        best_Q = res.x[1]

    #    print(best_sigma0, best_fc, best_Q)
        model_for_plot = model(freq_log, best_fc, best_sigma0, Q=best_Q, traveltime=traveltime, 
                               n=2, gamma=1,                       
                               inc_attenuation=True, inc_site_amplification=False,
                               inc_geometric_spreading=False, distance=hyp_dist/1000.)
        # plt.loglog(freq_log[SNR>3], model_for_plot[SNR>3], 'k-', lw=2, label='model')
        # plt.legend(loc=0)

        if best_fc < np.min(freq_log[SNR>3]):
            continue

        # sigma0 at source
        source_sigma0 =  best_sigma0

        m0 = moment(source_sigma0, hyp_dist, beta=beta, phase=phase)
        R = radius(best_fc, beta=beta, Vr=Vr, phase=phase)
        R_error = [radius(best_fc+fc_error[1], beta=beta, Vr=Vr, phase=phase), 
                   radius(best_fc-fc_error[0], beta=beta, Vr=Vr, phase=phase)]
        mw = moment_magnitude(m0)
        dstress = stress_drop(m0, R)
        dstress_error = [stress_drop(m0, R_error[1]), 
                         stress_drop(m0, R_error[0])]
        moment_mags[i] = mw
        radii[i,:] = np.asarray([R]+R_error)
        stress_drops[i,:] = np.asarray([dstress]+dstress_error)
        corner_frequencies[i,:] = np.asarray([best_fc]+fc_error)
        Q_param[i] = best_Q
        moment_0[i] = m0
        
    #     output_data[uid][station][comp]['brune'] = (source_sigma0, best_fc, fc_error, best_Q)
    #     output_data[uid][station][comp]['radius'] = (R, R_error)
    #     output_data[uid][station][comp]['moment'] = (m0, mw)
    #     output_data[uid][station][comp]['stress_drop'] = (dstress, dstress_error)

    #    print()
    #    print('MOMENT ANALYSIS')
    #    print('M0\t:\t{0:3.2e} Nm'.format(m0))
    #    print('fc\t:\t{0:5.3f} Hz'.format(best_fc))
    #    print('Q\t:\t{0:3.0f}'.format(best_Q))
    #    print('Radius\t:\t{0:3.2f} m'.format(R))
    #    print('dstress\t:\t{0:9.8f} MPa'.format(dstress/1e6))
    #    print('Mw\t:\t{0:5.2f}'.format(mw))
    #    print('ML\t:\t{0:5.2f}'.format(mag))
        print('{0:02d}\t{1:5.2f} {2:9.8f} MPa {3:3.2f} km'.format(ev.uid, mw, dstress/1e6, R/1000))
        print('\t{0:5.3f} Hz {1:d}'.format(best_fc, len(freq_log[SNR>3])))

        # plt.figure(3)
        # plt.loglog(fc_list, final_costs, 'k-', label='Cost')
        # plt.loglog(new_freq_samples, new_cost_y, 'g-', lw=2, label='Minima Fit')
        # plt.axvline(best_fc, c='k', ls='--', label='Min Cost')
        # plt.axhline(1.1*min_cost, c='b', ls='--', label='10% Error')
        # plt.axhline(1.2*min_cost, c='b', ls=':', label='20% Error')
        # plt.xlabel('Corner Frequency / Hz')
        # plt.ylabel('Cost')
        # plt.legend(loc=0)
        # plt.title(str(ev.uid)+' '+station+' '+raw.stats.channel)
        # # plt.savefig(swarm+'/{3:s}/cost_func_{0:02d}_{1:s}-{2:s}.pdf'.format(uid, station, comp, plot_loc))
        # plt.show()
        # plt.close('all')
        

    # assemble into a dataframe
    df = pds.DataFrame({"ID" : [tr.id for tr in st],
                        "Phase" : phases,
                        "epi_distance" : epidists,
                        "hyp_distance" : hypdists,
                        "ML" : [ml]*len(st),
                        "Mw" : moment_mags,
                        "M0" : moment_0,
                        "Q"  : Q_param,
                        "stress_drop" : stress_drops[:,0],
                        "min_stress_drop" : stress_drops[:,1],
                        "max_stress_drop" : stress_drops[:,2],
                        "radius" : radii[:,0],
                        "min_radius" : radii[:,1],
                        "max_radius" : radii[:,2],
                        "corner_frequency" : corner_frequencies[:,0],
                        "min_corner_frequency" : corner_frequencies[:,1],
                        "max_corner_frequency" : corner_frequencies[:,2]})
    return df

def write_moment_magnitude(ev, info, path):
    # summarise the output using S wave results only
    sort = (info.Phase == "S") & ~np.isnan(info.corner_frequency)
    mw = np.mean(info.Mw[sort])
    mw_err = np.std(info.Mw[sort]) / np.sum(sort)
    m0 = np.mean(info.M0[sort])
    ml = np.mean(info.ML[sort])
    fc = np.mean(info.corner_frequency[sort])
    Q = np.mean(info.Q[sort])
    R = np.mean(info.radius[sort])

    out = pds.DataFrame({"EventID" : [ev.uid],
                        "Mw" : [mw],
                        "Mw_err" : [mw_err],
                        "ML" : [ml],
                        "M0" : [m0],
                        "CornerFrequency" : [fc],
                        "Q" : [Q],
                        "Radius" : [R]})
    os.makedirs(os.path.join(path, "mw_summary"), exist_ok=True)
    out.to_csv(os.path.join(path, "mw_summary", f"{ev.uid}.csv"), index=False)

def write_moment_magnitude_summary(ev, info, path):
    os.makedirs(os.path.join(path, "mw_data"), exist_ok=True)
    info.to_csv(os.path.join(path, "mw_data", f"{ev.uid}.csv"), index=False)

def main(events, qmpath, inv, marginal_window=1., 
        prefilt=None, water_level=60., remove_full_response=False,
        prepick=-.2, postpick=5., nfft=1024, use_model_time=False,
        beta=2400, Vr=.9):

    qmpath = os.path.join(qmpath, "locate")

    for _, located_event in events.iterrows():
        event = Event(marginal_window)
        event.uid = located_event.EventID
        event.otime = UTC(located_event.DT)
        event.add_spline_location(located_event[["X", "Y", "Z"]])

        waveform_data = WaveformData(UTC(), UTC())
        waveform_data.response_inv = inv
        waveform_data.prefilt = prefilt
        waveform_data.water_level = water_level
        waveform_data.remove_full_response = remove_full_response
        waveform_data.waveforms = read(os.path.join(qmpath, "raw_cut_waveforms", f"{event.uid}.m"))
        waveform_data.raw_waveforms = waveform_data.waveforms
        waveform_data.starttime = waveform_data.waveforms[0].stats.starttime
        waveform_data.endtime = waveform_data.waveforms[0].stats.endtime

        event.add_waveform_data(waveform_data)
        event.add_picks(pds.read_csv(
                            os.path.join(qmpath, "picks", 
                                f"{event.uid}.picks")
                            )
                        )


        momag_info = calculate_moment_magnitude(event, located_event.ML,
                                            (prepick, postpick), nfft,
                                            beta)

        write_moment_magnitude(event, momag_info, qmpath)
        write_moment_magnitude_summary(event, momag_info, qmpath)



if __name__ == "__main__":
    # run script

    parser = argument_parser()
    args = parser.parse_args()

    if args.all_events:
        evpath = os.path.join(
                        args.qm_run, "locate", 
                        "events")
        events = pds.concat([
                pds.read_csv(
                    os.path.join(evpath, f"{fname}")
                            ) for fname in os.listdir(evpath)
                            ])
    elif args.filelist:
        events = pds.read_csv(args.filelist)
    else:
        raise ValueError("Need either filelist or to define --all_events")

    events["CoaTime"] = events.DT
    # events["UID"] = events.EventID
    main(events, args.qm_run, read_inventory(args.inventory),
        marginal_window=args.marginal_window,
        water_level=args.water_level,
        prefilt=args.prefilt,
        remove_full_response=args.remove_full_response,
        prepick=args.prepick, postpick=args.postpick,
        nfft=args.nfft, use_model_time=args.use_model_time,
        beta=args.beta, Vr=args.rupture_velocity)

    


def paz_2_amplitude_value_of_freq_resp(paz, freq):
    """
    Returns Amplitude at one frequency for the given poles and zeros

    :param paz: Given poles and zeros
    :param freq: Given frequency

    The amplitude of the freq is estimated according to "Of Poles and
    Zeros", Frank Scherbaum, p 43.

    .. rubric:: Example

    >>> paz = {'poles': [-4.44 + 4.44j, -4.44 - 4.44j],
    ...        'zeros': [0 + 0j, 0 + 0j],
    ...        'gain': 0.4}
    >>> amp = paz_2_amplitude_value_of_freq_resp(paz, 1)
    >>> print(round(amp, 7))
    0.2830262
    """
    jw = complex(0, 2 * np.pi * freq)  # angular frequency
    fac = complex(1, 0)
    for zero in paz['zeros']:  # numerator
        fac *= jw - zero
    for pole in paz['poles']:  # denominator
        fac /= jw - pole
    return abs(fac) * paz['gain']


x = wa.times()
y = wa.data
peaks, _ = find_peaks(y, prominence=0.5 * np.max(np.abs(y)))
troughs, _ = find_peaks(-y, prominence=0.5 * np.max(np.abs(y)))
if len(peaks) > len(troughs):
    peaks = peaks[:len(troughs)]
elif len(peaks) < len(troughs):
    troughs = troughs[:len(peaks)]

# find max adjacent trough-peak
full_peak_1 = np.abs(y[peaks[:-1]] - y[troughs[1:]])
full_peak_2 = np.abs(y[peaks[1:]] - y[troughs[:-1]])
if np.max(full_peak_1) >= np.max(full_peak_2):
    pos = np.argmax(full_peak_1)
    peaks = peaks[:-1]
    troughs = troughs[1:]
    wa_full_amp = np.max(full_peak_1)
else:
    pos = np.argmax(full_peak_2)
    peaks = peaks[1:]
    troughs = troughs[:-1]
    wa_full_amp = np.max(full_peak_2)

wa_t1 = wa.stats.starttime + wa.times()[peaks[pos]]
wa_t2 = wa.stats.starttime + wa.times()[troughs[pos]]
wa_time = [wa_t1, wa_t2]


x = raw.times()
y = raw.data
peaks, _ = find_peaks(y, prominence=0.5 * np.max(np.abs(y)))
troughs, _ = find_peaks(-y, prominence=0.5 * np.max(np.abs(y)))
if len(peaks) > len(troughs):
    peaks = peaks[:len(troughs)]
elif len(peaks) < len(troughs):
    troughs = troughs[:len(peaks)]

# find max adjacent trough-peak
full_peak_1 = np.abs(y[peaks[:-1]] - y[troughs[1:]])
full_peak_2 = np.abs(y[peaks[1:]] - y[troughs[:-1]])
if np.max(full_peak_1) >= np.max(full_peak_2):
    pos = np.argmax(full_peak_1)
    peaks = peaks[:-1]
    troughs = troughs[1:]
    raw_full_amp = np.max(full_peak_1)
else:
    pos = np.argmax(full_peak_2)
    peaks = peaks[1:]
    troughs = troughs[:-1]
    raw_full_amp = np.max(full_peak_2)

raw_t1 = raw.stats.starttime + raw.times()[peaks[pos]]
raw_t2 = raw.stats.starttime + raw.times()[troughs[pos]]
raw_time = [raw_t1, raw_t2]

# plt.plot(raw.data, 'k-')
# plt.plot(peaks, y[peaks], 'r+')
# plt.plot(troughs, y[troughs], 'b+')

# # contour_heights = y[peaks] - prominences
# # plt.vlines(x=peaks, ymin=contour_heights, ymax=y[peaks])

# # contour_heights = y[troughs] + depths
# # plt.vlines(x=troughs, ymin=y[troughs], ymax=contour_heights)

# plt.plot(peaks[pos], y[peaks[pos]], 'ro', ms=5)
# plt.plot(troughs[pos], y[troughs[pos]], 'bo', ms=5)
# plt.vlines(x=peaks, ymin=y[troughs], ymax=y[peaks])
# plt.show()

print(wa_full_amp)
print(raw_full_amp)
raw_freq = np.abs(raw_time[0] - raw_time[1]) * 2.
raw_freq = 1. / raw_freq
print(raw_freq)

plt.figure()

raw = st_raw[0]
# raw.filter('lowpass', freq=2.)
wa = st_wa[0]
print(raw, wa)

my_WOODANDERSON = WOODANDERSON
# my_WOODANDERSON['zeros'] = [0j]

raw_mag = np.abs(raw.stats.response.get_evalresp_response_for_frequencies([raw_freq], output='DISP'))
wa_mag = paz_2_amplitude_value_of_freq_resp(my_WOODANDERSON, raw_freq) * my_WOODANDERSON['sensitivity']

print((raw_full_amp * wa_mag) / raw_mag)

plt.plot(raw.times(), (raw.data / raw_mag) * wa_mag, 'k-')
plt.plot(wa.times(), wa.data, 'r--')
plt.axvline(tr.times()[1000])
plt.axvline(tr.times()[2000])

plt.show()


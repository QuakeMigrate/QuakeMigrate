Generate synthetic waveforms
============================

Now that we have a traveltime lookup table, we can use it to build realistic simulated waveforms for our example. Waveform simulation is a vast topic—here, we have opted for a relatively simple implementation, as it is sufficient for the purpose of walking through each QuakeMigrate stage. The earthquake source is represented as a pair of Gaussian-derivative wavelets (one for each of the primary body phases, P and S), with some given dominant frequency. Then, for each station, we use the pre-computed traveltimes for each phase to time-shift these Gaussian wavelets. Noise, drawn from a Gaussian distribution, is added to both the waveform amplitudes—nominally to represent background seismic noise—and the traveltimes—as a proxy for uncertainties in the velocity model. As with when we built the lookup table, we use a fixed seed to generate our random numbers to ensure repeatability between runs.

The wavelets are represented using a light dataclass, with a base class provided for users interested in exploring alternative wavelets:

.. code-block:: python

    @dataclass
    class Wavelet:
        """Light utility class to encapsulate a Wavelet."""

        sps: int
        time: np.ndarray
        data: np.ndarray

        def project(self, source_polarisation: float) -> tuple[np.ndarray, np.ndarray]:
            """
            Projects the wavelet onto a source polarisation.

            Parameters
            ----------
            source_polarisation: Source polarisation in degrees.

            Returns
            -------
            wavelet_x: Projected x-component of wavelet function.
            wavelet_y: Projected y-component of wavelet function.

            """

            wavelet_x = self.wavelet * np.cos(np.deg2rad(source_polarisation))
            wavelet_y = self.wavelet * np.sin(np.deg2rad(source_polarisation))

            return wavelet_x, wavelet_y


    class GaussianDerivativeWavelet(Wavelet):
        def __init__(self, frequency: float, sps: int, half_timespan: float) -> None:
            """Instantiate GaussianDerivativeWavelet object."""

            delta_T = 1 / frequency
            sigma = delta_T / 6
            self.frequency = frequency
            self.sps = sps

            half_timespan += 2 * delta_T

            self.time = np.arange(-half_timespan, half_timespan + 1 / sps, 1 / sps)
            data = (
                -self.time
                * np.exp(-(self.time**2) / (2 * sigma**2))
                / (sigma**3 * np.sqrt(2 * np.pi))
            )

            # Roll wavelet so first motion is at ~midpoint of array
            self.data = np.roll(data, int(sps * 0.5 / frequency) + 3) / max(data)

For simplicity, we place the earthquake at the center of our grid. The synthetics are computed on the LQT coordinate system, before being rotated onto the ZNE coordinate system to emulate the directional nature of the waveform propagation from the source to the instruments in the seismic network. That is, there is some partitioning of the P- and S-phase waveforms onto the vertical (Z) and horizontal (N and E) components, based on some angle-of-incidence (fixed at 80° from horizontal, in this instance), as well as the azimuth from the source to receiver. Finally, we scale the amplitudes of the phase waveforms using a simple Hutton-Boore local magnitude scaling relationship. The magnitude of the earthquake can be varied to explore how QuakeMigrate performs at different scales (though, in reality, this would also be reflected by a corresponding change in the frequency content of the waveforms, which is not addressed here).

Here, we step through the underlying function within the small :mod:`simulate` module found in the example directory, which is used to simulated waveforms, starting with the function header:

.. code-block:: python

    def simulate_waveforms(
        wavelet: Wavelet,
        earthquake_coords: tuple[float, float, float],
        lut: LUT,
        magnitude: int = 1,
        noise: dict | None = None,
        angle_of_incidence: int = 0,
    ) -> Stream:
        """
        Simulates the waveforms expected for an earthquake within a given LUT.

        Performs simulation in LQT-space (Latitudinal, SV direction, SH direction), before
        rotating onto ZNE based on the ray angles (back-azimuth and inclination).

        Parameters
        ----------
        wavelet: The base wavelet used to represent the waveform for each simulated phase.                                                                 
        earthquake_coords: The lon, lat, and depth of the earthquake.
        lut: A QuakeMigrate traveltime lookup table, used to migrate simulated waveforms.
        magnitude: A local magnitude used to simulate the effect of distance attenuation.
        noise: Gaussian noise scaling for simulated waveform traveltimes and amplitudes.
        angle_of_incidence: Used to rotate from LQT onto ZNE axes.

        Returns
        -------
        stream: An ObsPy Stream object containing the simulated waveform traces.

        """

The function requires 3 inputs: the base wavelet, the coordinates of the earthquake (longitude, latitude, depth), and the traveltime lookup table. It also takes 3 optional inputs: an earthquake magnitude, Gaussian noise scaling factors, and angle-of-incidence. For each of these optional inputs, some default values are provided.

Then, for each station, the synthetics are built. Below, we have added additional annotations (indicated by the lines starting with ``#!``) to the code within the for-loop block, in order to highlight each stage of the process discussed above.

.. code-block:: python

    station = station_data["Name"]

    #! Compute the distance and azimuth between the source and receiver
    hypo_dist, az, baz = _gps2hypodist_az_baz(
        station_data, earthquake_coords, lut.unit_conversion_factor
    )

    #! Compute the amplitude decay factor as a function of distance
    amp_factor = 10 ** (magnitude - _attenuate(hypo_dist))

    # Build L component, e.g. the P-phase synthetic
    P = Trace()
    #! Find the traveltime from the earthquake to the given station
    P_ttime = lut.traveltime_to("P", earthquake_ijk, station=station)
    #! Add some Gaussian noise to this traveltime
    P_ttime += np.random.normal(scale=noise["traveltime"]["P"], size=1)
    #! Compute the number of samples by which to shift the waveforms
    roll_by = int(wavelet.sps * P_ttime)
    #! Compute the Gaussian noise to be added to the waveform amplitude
    P_amp_noise = np.random.normal(
        scale=noise["amplitude"]["P"], size=len(wavelet.data)
    )
    #! Time shift the wavelet after scaling the amplitude and adding amplitude noise
    P.data = np.roll(wavelet.data.copy() * amp_factor * 0.5 + P_amp_noise, roll_by)

    # Build Q/T components, e.g. the S-phase synthetic
    S1, S2 = Trace(), Trace()
    S_ttime = lut.traveltime_to("S", earthquake_ijk, station=station)
    S_ttime += np.random.normal(scale=noise["traveltime"]["S"], size=1)
    roll_by = int(wavelet.sps * S_ttime)
    S_amp_noise = np.random.normal(
        scale=noise["amplitude"]["S"], size=len(wavelet.data)
    )
    S1.data = np.roll(wavelet.data.copy() * amp_factor + S_amp_noise, roll_by)
    #! The process is the same as for the P-phase, except we need to split the
    #! horizontal component onto the Q and T axes
    S2.data = np.zeros(len(S1.data)) + S_amp_noise

    #! Create an ObsPy Stream object and specify some metadata
    lqt_stream = Stream()
    for component, trace in zip("LQT", [P, S1, S2]):
        trace.stats.starttime = UTCDateTime("2021-02-18T12:00:00.0")
        trace.stats.sampling_rate = wavelet.sps
        trace.stats.station = station
        trace.stats.network = "SC"
        trace.stats.channel = f"CH{component}"
        lqt_stream += trace

    #! Take advantange of the ObsPy stream rotation utilities to transform from
    #! LQT to ZNE
    zne_stream = lqt_stream.rotate(
        "LQT->ZNE", back_azimuth=baz, inclination=inclination
    )

    #! Add the new stream to the collection of streams
    stream += zne_stream

Once run, a set of miniSEED files will be written to the ``inputs/mSEED`` directory, which a file for each station/component pair.

The full script looks like this:

.. code-block:: python

    """
    This script generates synthetic waveforms to accompany the tutorial in the online
    documentation. 

    :copyright:
        2020–2024, QuakeMigrate developers.
    :license:
        GNU General Public License, Version 3
        (https://www.gnu.org/licenses/gpl-3.0.html)

    """

    import pathlib

    import numpy as np
    from quakemigrate.io import read_lut

    import simulate as simulate


    lut = read_lut("./outputs/lut/example.LUT")

    mseed_output_dir = pathlib.Path.cwd() / "inputs/mSEED/2021/049"
    mseed_output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate synthetic wavelets and migrate by calculated traveltimes
    np.random.seed(4)  # Fix seed for reproducible results

    # --- Build wavelet ---
    frequency, sps, time_span = 4.0, 100, 300.0
    wavelet = simulate.GaussianDerivativeWavelet(frequency, sps, time_span)

    earthquake_coords = [0.0, 0.0, 15.0]
    aoi = 80
    magnitude = 2.2

    simulated_stream = simulate.simulate_waveforms(
        wavelet, earthquake_coords, lut, magnitude=magnitude, angle_of_incidence=aoi
    )

    for tr in simulated_stream:
        fname = f"inputs/mSEED/2021/049/{tr.stats.station}_{tr.stats.component}.m"
        tr.write(fname, format="MSEED")

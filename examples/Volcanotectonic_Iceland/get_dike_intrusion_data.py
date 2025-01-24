# -*- coding: utf-8 -*-
"""
This script will download the waveform data and an instrument response
inventory from IRIS (in miniSEED and STATIONXML formats, respectively)
for the Iceland dike intrusion example.

"""

import pathlib

from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import (
    GlobalDomain,
    Restrictions,
    MassDownloader,
)

from quakemigrate.io import read_stations

# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
data_path = pathlib.Path("./inputs/mSEED")
stationxml_storage = "./inputs/DATALESS"


# --- Define directory structure for storing waveform data ---
def get_mseed_storage(network, station, location, channel, starttime, endtime):
    fname = (
        data_path
        / f"{starttime.year}"
        / f"{starttime.julday:03d}"
        / f"{station}_{channel[2]}.m"
    ).as_posix()

    return fname


# --- Set network code & client ---
network = "Z7"
datacentres = ["IRIS"]
# global domain (specifying network and stations instead)
domain = GlobalDomain()

# --- Set time period over which download data ---
starttime = UTCDateTime("2014-236T00:00:00")
endtime = UTCDateTime("2014-236T00:15:00")

# --- Read in station file ---
stations = read_stations(station_file)
stations_string = ",".join(stations["Name"])

# --- Set up request ---
restrictions = Restrictions(
    starttime=starttime,
    endtime=endtime,
    chunklength_in_sec=86400,
    network=network,
    station=stations_string,
    channel_priorities=["HH[ZNE]", "BH[ZNE]"],
    minimum_interstation_distance_in_m=0,
)

# --- Download waveform data ---
mdl = MassDownloader(providers=datacentres)
mdl.download(
    domain,
    restrictions,
    threads_per_client=3,
    mseed_storage=get_mseed_storage,
    stationxml_storage=stationxml_storage,
)

# -*- coding: utf-8 -*-
"""
This script will download the waveform data and instrument response inventory
from IRIS (in miniSEED and STATIONXML formats, respectively) for the Iceland
dike intrusion example.

"""

import pathlib

from obspy import UTCDateTime, Inventory
from obspy.clients.fdsn import Client

from quakemigrate.io import read_stations

# --- i/o paths ---
station_file = "./inputs/iceland_stations.txt"
data_path = pathlib.Path("./inputs/mSEED")
response_file = "./inputs/Z7_dataless.xml"

# --- Set network code & client ---
network = "Z7"
datacentre = "IRIS"
client = Client(datacentre)

# --- Set time period over which download data ---
starttime = UTCDateTime("2014-236T00:00:00")
endtime = UTCDateTime("2014-236T00:15:00")

#  --- Read in station file ---
stations = read_stations(station_file)

# --- Download instrument response inventory ---
inv = Inventory()
for station in stations["Name"]:
    inv += client.get_stations(network=network, station=station,
                               starttime=starttime, endtime=endtime,
                               level="response")
inv.write(response_file, format="STATIONXML")

# --- Make directories to store waveform data ---
waveform_path = data_path / str(starttime.year) / f"{starttime.julday:03d}"
waveform_path.mkdir(parents=True, exist_ok=True)

# --- Download waveform data ---
for station in stations["Name"]:
    print(f"Downloading waveform data for station {station} from {datacentre}")
    st = client.get_waveforms(network=network, station=station,
                              location="*", channel="*H*",
                              starttime=starttime, endtime=endtime)
    st.merge(method=-1)
    for comp in ["E", "N", "Z"]:
        try:
            st_comp = st.select(component=comp)
            st_comp.write(str(waveform_path / f"{station}_{comp}.m"),
                          format="MSEED")
        except IndexError:
            pass

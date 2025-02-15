"""
This script will download the waveform data and an instrument response inventory from
EarthScope (in miniSEED and STATIONXML formats, respectively) for the Ethiopia swarm
example.

:copyright:
    2020–2025, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

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
station_file = "./inputs/ethiopia_stations_TM.csv"
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
network = "Y6"
datacentres = ["IRIS"]
# global domain (specifying network and stations instead)
domain = GlobalDomain()

# --- Set time period over which download data ---
starttime = UTCDateTime("2016-097T18:30:00.0")
endtime = UTCDateTime("2016-097T18:45:00.0")

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
    channel_priorities=["BH[ZNE]"],
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


# from obspy.clients.fdsn import Client
# from obspy import UTCDateTime as UTC
# from quakemigrate.util import inventory_to_QM
# import os

# client = Client('IRIS')

# inv = client.get_stations(network="Y6", minlatitude=7.85, maxlatitude=8.3,
#         starttime=UTC('2016-097T'), endtime=UTC('2016-098T'),
#         level='response')

# inventory_to_QM(inv, outputfile='./inputs/ethiopia_stations_TM.csv')
# inv.write('./inputs/Y6.dataless.xml', format='STATIONXML')

# starttime = UTC("2016-097T18:30:00")
# endtime = UTC("2016-097T18:45:00")

# try:
#     os.makedirs('inputs/mSEED/2016/097')
# except:
#     pass

# for network in inv:
#     for station in network:
#         print('Getting', network.code, station.code, 'from IRIS...')
#         try:
#             st = client.get_waveforms(network.code, station.code, "*", "BH?",
#                                     starttime, endtime)
#         except:
#             print('\t...No data')
#             continue

#         print('\t...Writing miniSEED')
#         year = st[0].stats.starttime.year
#         jday = st[0].stats.starttime.julday
#         for tr in st:
#             tr.write('inputs/mSEED/{}/{:03d}/{}.mseed'.format(year, jday, tr.id),
#                     format='MSEED')

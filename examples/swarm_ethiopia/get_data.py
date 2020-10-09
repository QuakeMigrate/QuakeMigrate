from obspy.clients.fdsn import Client
from obspy import UTCDateTime as UTC
from quakemigrate.util import inventory_to_QM
import os

client = Client('IRIS')

inv = client.get_stations(network="Y6", minlatitude=7.85, maxlatitude=8.3,
        starttime=UTC('2016-097T'), endtime=UTC('2016-098T'),
        level='response')

inventory_to_QM(inv, outputfile='./inputs/ethiopia_stations_TM.csv')
inv.write('./inputs/Y6.dataless.xml', format='STATIONXML')

starttime = UTC("2016-097T18:30:00")
endtime = UTC("2016-097T18:45:00")

try:
    os.makedirs('inputs/mSEED/2016/097')
except:
    pass

for network in inv:
    for station in network:
        print('Getting', network.code, station.code, 'from IRIS...')
        try:
            st = client.get_waveforms(network.code, station.code, "*", "BH?", 
                                    starttime, endtime)
        except:
            print('\t...No data')
            continue

        print('\t...Writing miniSEED')
        year = st[0].stats.starttime.year
        jday = st[0].stats.starttime.julday
        for tr in st:
            tr.write('inputs/mSEED/{}/{:03d}/{}.mseed'.format(year, jday, tr.id),
                    format='MSEED')




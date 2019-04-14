############################################################################
############## Scripts for Scanning and Coalescence of Data ################
############################################################################
# ---- Import Packages -----
import obspy
from obspy import UTCDateTime
import QMigrate.core.model as cmod


from datetime import datetime
from datetime import timedelta
from glob import glob

import numpy as np


# ----- Useful Functions -----
def _downsample(st,sr):
    '''
        Downsampling the MSEED to the designated sampling rate

        Add Error handeling of the  decimate of non-int

    '''
    for i in range(0,len(st)):
        #st[i].decimate(factor=int(st[i].stats.sampling_rate/sr), strict_length=False)
        if sr != st[i].stats.sampling_rate:
                st[i].filter('lowpass',freq=float(sr) / 2.000001,corners=2,zerophase=True)
                st[i].decimate(factor=int(st[i].stats.sampling_rate/sr), strict_length=False, no_filter=True)

    return st



class MSEED():

    def __init__(self,LUT,HOST_PATH='/PATH/MSEED'):
        self.startTime           = None
        self.endTime             = None
        self.sampling_rate       = None
        self.MSEED_path          = HOST_PATH

        self.Type                = None 
        self.FILES               = None 
        self.signal              = None
        self.FilteredSignal      = None
        self.StationAvaliability = None

        lut = cmod.LUT()
        lut.load(LUT)
        self.StationInformation  = lut.station_data
        del lut
        self.st                  = None


    def _stationAvaliability(self,st):
        ''' Reading the Avaliability of the stations between two times 

        '''
        stT  = self.startTime
        endT = self.endTime

        # Since the traces are the same sample-rates then the stations can be selected based
        #on the start and end time
        exSamples = round((endT-stT).total_seconds()*self.sampling_rate + 1)

        stationAva = np.zeros((len(self.StationInformation['Name']),1))
        signal     = np.zeros((3,len(self.StationInformation['Name']),int(exSamples)))

        for i in range(0,len(self.StationInformation['Name'])):

            tmp_st = st.select(station=self.StationInformation['Name'][i])
            if len(tmp_st) == 3:
                if tmp_st[0].stats.npts <= exSamples and tmp_st[1].stats.npts == exSamples and tmp_st[2].stats.npts == exSamples:
                    # Defining the station as avaliable
                    stationAva[i] = 1
                    
                    for tr in tmp_st:
                        # Giving each component the correct signal
                        if tr.stats.channel[-1] == 'E' or tr.stats.channel[-1] == '2':
                            signal[1,i,:] = tr.data

                        if tr.stats.channel[-1] == 'N' or tr.stats.channel[-1] == '1':
                            signal[0,i,:] = tr.data

                        if tr.stats.channel[-1] == 'Z':
                            signal[2,i,:] = tr.data

            else:
                # Trace not completly active during this period
                continue 

        return signal,stationAva


    def path_structure(self,TYPE='YEAR/JD/STATION'):
        ''' Function to define the path structure of the mseed. 

            This is a complex problem and will depend entirely on how the data is structured.
            Since the reading of the headers is quick we only need to get to the write data.
        '''

        if TYPE == 'YEAR/JD/STATION':
            self.Type  = 'YEAR/JD/STATION'

        if TYPE == 'STATION.YEAR.JULIANDAY':
            self.Type = 'STATION.YEAR.JULIANDAY'

        if TYPE == '/STATION/STATION.YearMonthDay':
            self.Type = '/STATION/STATION.YearMonthDay'


    def _load_fromPath(self):
        '''
            Given the type of path structure load the data in the required format

        '''
        if self.Type == None:
            print('Please Specfiy the path_structure - DATA.path_structure')
            return

        if self.Type == 'YEAR/JD/STATION':
            dy = 0
            FILES = []
            #print(float(self.endTime.year) + float('0.{}'.format(self.endTime.timetuple().tm_yday)))
            #print(float(self.startTime.year) + float('0.{}'.format((self.startTime + timedelta(days=dy)).timetuple().tm_yday)))
            while self.endTime.timetuple().tm_yday >=  (self.startTime + timedelta(days=dy)).timetuple().tm_yday:
                # Determine current time
                ctime = self.startTime + timedelta(days=dy)
                #print(ctime)
                for st in self.StationInformation['Name'].tolist():
                    FILES.extend(glob('{}/{}/{}/*{}*'.format(self.MSEED_path,ctime.year,str(ctime.timetuple().tm_yday).zfill(3),st)))

                dy += 1 

            FILES = set(FILES)

        if self.Type == 'STATION.YEAR.JULIANDAY':
            dy = 0
            FILES = []
            #print(float(self.endTime.year) + float('0.{}'.format(self.endTime.timetuple().tm_yday)))
            #print(float(self.startTime.year) + float('0.{}'.format((self.startTime + timedelta(days=dy)).timetuple().tm_yday)))
            while self.endTime >=  (self.startTime + timedelta(days=dy)):
                # Determine current time
                ctime = self.startTime + timedelta(days=dy)
                #print(ctime)
                for st in self.StationInformation['Name'].tolist():
                    FILES.extend(glob('{}/*{}.*.{}.{}'.format(self.MSEED_path,st,ctime.year,str(ctime.timetuple().tm_yday).zfill(3))))

                dy += 1 

        if self.Type == '/STATION/STATION.YearMonthDay':
            dy = 0
            FILES = []
            while self.endTime >=  (self.startTime + timedelta(days=dy)):
                # Determine current time
                ctime = self.startTime + timedelta(days=dy)
                #print(ctime)
                for st in self.StationInformation['Name'].tolist():
                    #print('{}/{}/{}.{}{:02d}{:02d}*'.format(self.MSEED_path,st,str(st).lower(),str(ctime.year)[-2:],ctime.month,ctime.day))

                    FILES.extend(glob('{}/{}/{}.{}{:02d}{:02d}*'.format(self.MSEED_path,st,str(st).lower(),str(ctime.year)[-2:],ctime.month,ctime.day)))
                dy += 1
            #print(FILES)
        self.FILES = FILES


    def read_mseed(self,starttime,endtime,sampling_rate):
        ''' 
            Reading the required mseed files for all stations between two times and return 
            station avaliability of the seperate stations during this period
        '''


        self.startTime = datetime.strptime(starttime,'%Y-%m-%dT%H:%M:%S.%f')
        self.endTime      = datetime.strptime(endtime,'%Y-%m-%dT%H:%M:%S.%f')
        self.sampling_rate = sampling_rate
        self._load_fromPath()

        if len(self.FILES) > 0:
		# Loading the required mseed data
                c=0
                for f in self.FILES:
                  try:
                    if c==0:
                      self.st     = obspy.read(f,starttime=UTCDateTime(self.startTime),endtime=UTCDateTime(self.endTime))
                      self.st_org = obspy.read(f,starttime=UTCDateTime(self.startTime),endtime=UTCDateTime(self.endTime))
                      c +=1
                    else:
                      self.st += obspy.read(f,starttime=UTCDateTime(self.startTime),endtime=UTCDateTime(self.endTime))
                      self.st_org += obspy.read(f,starttime=UTCDateTime(self.startTime),endtime=UTCDateTime(self.endTime))
                  except:
                    continue
                    print('Station File not MSEED - {}'.format(f))

                # Removing all the stations with gaps greater than 10.0 milliseconds
                #print(self.st)
                self.st._cleanup()
                if len(self.st.get_gaps()) > 0:
                  stationRem = np.unique(np.array(self.st.get_gaps())[:,1]).tolist()
                  for sa in stationRem:
                    tr = self.st.select(station=sa)
                    for tra in tr:
                      self.st.remove(tra) 


                # Combining the mseed and determining station avaliability
                self.st.detrend('linear')
                self.st.detrend('demean')
                self.st = _downsample(self.st,sampling_rate)
                signal,stA = self._stationAvaliability(self.st)

        else:
                print('Data Does not exist for this time period - creating blank')
                # Files don't exisit so creating zeros ones instead
                exSamples = (endT-stT).total_seconds()*self.sampling_rate + 1
                stationAva = np.zeros((len(self.StationInformation['Name']),1))
                signal     = np.zeros((3,len(self.StationInformation['Name']),int(exSamples)))


#        self.st      = None
        self.signal  = signal
        self.FilteredSignal = np.empty((self.signal.shape))
        self.FilteredSignal[:] = np.nan
        self.station_avaliability = stA



################################################################################################



# ---- Import Packages -----
import matplotlib
import numpy as np
from QMigrate.core.time import UTCDateTime
import QMigrate.core.model as cmod
from datetime import datetime
from datetime import timedelta

from obspy import read,Stream,Trace
from obspy.core import UTCDateTime
from obspy.signal.invsim import cosine_taper

from obspy.signal.trigger import classic_sta_lta
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
from scipy import stats

import QMigrate.core.QMigratelib as ilib

import obspy
import re

import os
import os.path as path
import pickle

import pandas as pd

try:
    os.environ['DISPLAY']
    matplotlib.use('Qt4Agg')
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pylab as plt


from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from matplotlib import patches
import matplotlib.image as mpimg
import matplotlib.animation as animation



# Defining the machine learning packages 
import string
import time
import argparse as ap
import sys
import os

# import numpy as np
# import obspy
# import obspy.core as oc
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv1D, MaxPooling1D
# from keras import losses
# from keras.models import model_from_json
# import tensorflow as tf

# ----- Timing functions -----

import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)



# ----- Useful Functions -----


def gaussian_func(x,a,b,c):
    '''


    '''
    f = a*np.exp(-1.*((x-b)**2)/(2*(c**2)))
    return f


def sta_lta_centred(a, nsta, nlta):
    '''


    '''

    # Forcing nsta and nlta to be intergers
    nsta = int(nsta)
    nlta = int(nlta)

    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(a ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    #sta[:-nsta] = sta[nsta:] - sta[:-nsta] 
    sta[nsta:] = sta[nsta:] - sta[:-nsta]  
    sta[nsta:-nsta] = sta[nsta*2:]
    sta /= nsta

    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:(nlta -1)] = 0
    sta[-nsta:] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta/ lta



def onset(sig, stw, ltw,centred=False):
    '''


    '''
    # assert isinstance(snr, object)
    nchan, nsamp = sig.shape
    snr = np.copy(sig)
    snr_raw = np.copy(sig)
    for ch in range(0, nchan):
        if np.sum(sig[ch,:]) == 0.0:
            snr[ch, :] = 0.0
            snr_raw[ch, :] = snr[ch,:]
        else:
            if centred == True:
                snr[ch, :] = sta_lta_centred(sig[ch, :], stw, ltw)
            else:
                snr[ch, :] = classic_sta_lta(sig[ch, :], stw, ltw)
            snr_raw[ch, :] = snr[ch,:]
            np.clip(1+snr[ch,:],1.0,np.inf,snr[ch, :])
            np.log(snr[ch, :], snr[ch, :])

    return snr_raw,snr


def filter(sig,srate,lc,hc,order=3):
    '''


    '''
    b1, a1 = butter(order, [2.0*lc/srate, 2.0*hc/srate], btype='band')
    nchan, nsamp = sig.shape
    fsig = np.copy(sig)
    #sig = detrend(sig)
    for ch in range(0, nchan):
        fsig[ch,:] = fsig[ch,:] - fsig[ch,0]

        tap = cosine_taper(len(fsig[ch,:]),0.1)
        fsig[ch,:] = fsig[ch,:]*tap
        fsig[ch,:] = lfilter(b1, a1, fsig[ch,::-1])[::-1]
        fsig[ch,:] = lfilter(b1, a1, fsig[ch,:])
    return fsig



def _find(obj, name, default=None):
    if isinstance(name, str):
        if name in obj:
            return obj[name]
        else:
            return default
    elif name[0] in obj:
        if len(name) == 1:
            return obj[name[0]]
        else:
            return _find(obj[name[0]], name[1:], default)
    else:
        return default


def _read_scan(fname):
    CoaVal = pd.read_csv(fname,names=['DT','COA','X','Y','Z'])
    CoaVal['DT'] = pd.to_datetime(CoaVal['DT'])
    return CoaVal





class SeisOutFile:
    '''
        Definition of manipulation types for the Seismic scan files.

    '''

    def __init__(self, path = '', name = None):
        self.open(path, name)
        self.FileSampleRate = None #Sample rate in miliseconds

    def open(self, path = '', name = None):
        self.path = path
        if name is None:
            name = datetime.now().strftime('RUN_%Y%m%d_%H%M%S')
        self.name = name
        print('Path = ' + repr(self.path) + ', Name = ' + repr(self.name))


    def read_scan(self):
        fname = path.join(self.path,self.name + '.scn')
        DATA = _read_scan(fname)
        return DATA

    def read_coal4D(fname):
        map = np.load(fname)
        return map


    def read_decscan(self):
        fname = path.join(self.path,self.name + '.scnmseed')
        COA = obspy.read(fname)

        sampling_rate = COA.select(station='COA')[0].stats.sampling_rate

        DATA = pd.DataFrame()


        DATA['DT']       = np.arange(datetime.strptime(str(COA.select(station='COA')[0].stats.starttime),'%Y-%m-%dT%H:%M:%S.%fZ'),datetime.strptime(str(COA.select(station='COA')[0].stats.endtime),'%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(seconds=1/COA.select(station='COA')[0].stats.sampling_rate) ,timedelta(seconds=1/COA.select(station='COA')[0].stats.sampling_rate))
        DATA['COA']      = COA.select(station='COA')[0].data/1e8
        DATA['COA_NORM'] = COA.select(station='COA_N')[0].data/1e8
        DATA['X']        = COA.select(station='X')[0].data/1e6
        DATA['Y']        = COA.select(station='Y')[0].data/1e6
        DATA['Z']        = COA.select(station='Z')[0].data

        DATA['DT'] = pd.to_datetime(DATA['DT'])

        return DATA
        
    def read_events(self,starttime,endtime):
        fname = path.join(self.path,self.name + '_TriggeredEvents.csv')
        EVENTS = pd.read_csv(fname)

        # Trimming the events to the time in question
        EVENTS['CoaTime'] = pd.to_datetime(EVENTS['CoaTime'])
        EVENTS = EVENTS[(EVENTS['CoaTime'] >= datetime.strptime(starttime,'%Y-%m-%dT%H:%M:%S.%f')) & (EVENTS['CoaTime'] <= datetime.strptime(endtime,'%Y-%m-%dT%H:%M:%S.%f'))]

        #  Changing the columns from string to datetimes
        EVENTS['MinTime'] = pd.to_datetime(EVENTS['MinTime'])
        EVENTS['MaxTime'] = pd.to_datetime(EVENTS['MaxTime'])

        return EVENTS



    def write_log(self, message):
        fname = path.join(self.path,self.name + '.log')
        with open(fname, "a") as fp:
            fp.write(message + '\n')




    def cut_mseed(self,DATA,EventName):
        fname = path.join(self.path,self.name + '_{}.mseed'.format(EventName))
        
        st = DATA.st
        st.write(fname, format='MSEED')



    def del_scan(self):
        fname = path.join(self.path,self.name + '.scn')
        if path.exists(fname):
           print('Filename {} already exists. Deleting !'.format(fname))
           os.system('rm {}'.format(fname))


    def write_scan(self,daten,dsnr,dsnr_norm,dloc):
        # Defining the ouput filename
        fname = path.join(self.path,self.name + '.scn')

        # Defining the array to save
        ARRAY = np.array((daten,dsnr,dloc[:,0],dloc[:,1],dloc[:,2]))
        # # if 
        if self.FileSampleRate == None:
            DF             = pd.DataFrame(columns=['DT','COA','COA_NORM','X','Y','Z'])
            DF['DT']       = daten
            DF['DT']       = pd.to_datetime(DF['DT'])
            DF['DT']       = DF['DT'].astype(str)
            DF['COA']      = dsnr
            DF['COA_NORM'] = dsnr_norm
            DF['X']        = dloc[:,0]
            DF['Y']        = dloc[:,1]
            DF['Z']        = dloc[:,2]

        else:
            # Resampling the data on save
            DF = pd.DataFrame(columns=['DT','COA','COA_NORM','X','Y','Z'])
            DF['DT']  = daten
            DF['COA'] = dsnr
            DF['COA_NORM'] = dsnr_norm
            DF['X']   = dloc[:,0]
            DF['Y']   = dloc[:,1]
            DF['Z']   = dloc[:,2]
            DF['DT'] = pd.to_datetime(DF['DT'])
            #DF = DF.set_index(pd.DatetimeIndex(DF['DT']))
            DF = DF.set_index(DF['DT'])
            DF = DF.resample('{}L'.format(self.FileSampleRate)).mean()
            DF = DF.reset_index()
            DF = DF.rename(columns={"index":"DT"})
            DF['DT'] = DF['DT'].astype(str)

        if path.exists(fname):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not

        ARRAY = np.array(DF)

        with open(fname, append_write) as fp:
            for ii in range(ARRAY.shape[0]):
                fp.write('{},{},{},{},{},{}\n'.format(ARRAY[ii,0],ARRAY[ii,1],ARRAY[ii,2],ARRAY[ii,3],ARRAY[ii,4],ARRAY[ii,5]))


    def write_events(self,EVENTS):
        fname = path.join(self.path,self.name + '_TriggeredEvents.csv')
        EVENTS.to_csv(fname,index=False)




    def write_decscan(self,OriginalST,daten,dsnr,dsnr_norm,dloc,sampling_rate):
        scn_fname = path.join(self.path,self.name +'.scn')
        mseed_fname = path.join(self.path,self.name +'.scnmseed')


        #            
        DATA             = pd.DataFrame(columns=['DT','COA','COA_NORM','X','Y','Z'])
        DATA['DT']       = daten
        DATA['DT']       = pd.to_datetime(DATA['DT'])
        DATA['COA']      = dsnr
        DATA['COA_NORM'] = dsnr_norm
        DATA['X']        = dloc[:,0]
        DATA['Y']        = dloc[:,1]
        DATA['Z']        = dloc[:,2]
        #print(DATA)

        # Turning the results into a MSEED format

        stats_COA      = {'network': 'NW', 'station':'COA','npts':len(DATA),'sampling_rate':sampling_rate}
        stats_COA_norm = {'network': 'NW', 'station':'COA_N','npts':len(DATA),'sampling_rate':sampling_rate}
        stats_X        = {'network': 'NW', 'station':'X','npts':len(DATA),'sampling_rate':sampling_rate}
        stats_Y        = {'network': 'NW', 'station':'Y','npts':len(DATA),'sampling_rate':sampling_rate}
        stats_Z        = {'network': 'NW', 'station':'Z','npts':len(DATA),'sampling_rate':sampling_rate}

        stats_COA['starttime']      = DATA.iloc[0][0]
        stats_COA_norm['starttime'] = DATA.iloc[0][0]
        stats_X['starttime']        = DATA.iloc[0][0]
        stats_Y['starttime']        = DATA.iloc[0][0]
        stats_Z['starttime']        = DATA.iloc[0][0]

        ST = Stream(Trace(data=(np.array(DATA['COA'])*1e8).astype(np.int32),header= stats_COA))
        ST = ST + Stream(Trace(data=(np.array(DATA['COA_NORM'])*1e8).astype(np.int32),header= stats_COA_norm))
        ST = ST + Stream(Trace(data=(np.array(DATA['X'])*1e6).astype(np.int32),header= stats_X))
        ST = ST + Stream(Trace(data=(np.array(DATA['Y'])*1e6).astype(np.int32),header= stats_Y))
        ST = ST + Stream(Trace(data=(np.array(DATA['Z'])).astype(np.int32),header= stats_Z))

        #print(ST.select(station='COA')[0].data/1e8)
        #print(ST.select(station='X')[0].data/1e6)
        #print(ST.select(station='Y')[0].data/1e6)
        #print(ST.select(station='Z')[0].data)

        # Appending to the orignal dataset if exists
        if OriginalST != None:    
            OriginalST = OriginalST + ST
        else:
            OriginalST = ST



        OriginalST.write(mseed_fname,format='MSEED',encoding=11)


        return OriginalST




    def write_coal4D(self,map4D,EVENT,stT,enT):
        cstart = datetime.strftime(stT,'%Y-%m-%dT%H:%M:%S.%f')
        cend   = datetime.strftime(enT,'%Y-%m-%dT%H:%M:%S.%f')
        fname = path.join(self.path,self.name + '{}_{}_{}.coal4D'.format(EVENT,cstart,cend))
        
        # This file size is massive ! write as binary and include load function

        # Define the X0,Y0,Z0,T0,Xsiz,Ysiz,Zsiz,Tsiz,Xnum,Ynum,Znum,Tnum
        np.save(fname,map4D)


    def write_coalVideo(self,MAP,lookup_table,DATA,EventCoaVal,EventName,AdditionalOptions=None):
        '''
            Writing the coalescence video to file for each event
        '''

        filename = path.join(self.path,self.name)

        SeisPLT = SeisPlot(MAP,lookup_table,DATA,EventCoaVal)
    
        SeisPLT.CoalescenceVideo(SaveFilename='{}_{}'.format(filename,EventName))
        SeisPLT.CoalescenceMarginal(SaveFilename='{}_{}'.format(filename,EventName))

    def write_stationsfile(self,STATION_pickS,EventName):
        '''
            Writing the stations file
        '''
        fname = path.join(self.path,self.name + '_{}.stn'.format(EventName))
        STATION_pickS.to_csv(fname,index=False)

    def write_event(self,EVENT,EventName):
        '''
            Writing the stations file
        '''
        fname = path.join(self.path,self.name + '_{}.event'.format(EventName))
        EVENT.to_csv(fname,index=False)


class SeisPlot:
    '''
         Seismic plotting for QMigrate ouptuts. Functions include:
            CoalescenceVideo - A script used to generate a coalescence 
            video over the period of earthquake location

            CoalescenceLocation - Location plot 

            CoalescenceMarginalizeLocation - 

    '''
    def __init__(self,lut,MAP,CoaMAP,DATA,EVENT,StationPick,MarginalWindow,PlotOptions=None):
        '''
            This is the initial variatiables
        '''
        self.LUT         = lut
        self.DATA        = DATA
        self.EVENT       = EVENT
        self.MAP         = MAP
        self.CoaMAP      = CoaMAP
        self.StationPick = StationPick
        self.RangeOrder  = True


        if PlotOptions == None:
            self.TraceScaling     = 1
            self.CMAP             = 'hot_r'
            self.LineStationColor = 'black'
            self.Plot_Stations    = True
            self.FilteredSignal   = True
            self.XYFiles          = None
        else:
            try:
                self.TraceScaling     = PlotOptions.TraceScaling
                self.CMAP             = PlotOptions.MAPColor
                self.LineStationColor = PlotOptions.LineStationColor
                self.Plot_Stations    = PlotOptions.Plot_Stations
                self.FilteredSignal   = PlotOptions.FilteredSignal
                self.XYFiles          = PlotOptions.XYFiles

            except:
                print('Error - Please define all plot option, see function ... for details.')



        self.times = np.arange(self.DATA.startTime,self.DATA.endTime,timedelta(seconds=1/self.DATA.sampling_rate))
        self.EVENT = self.EVENT[(self.EVENT['DT'] > self.times[0]) & (self.EVENT['DT'] < self.times[-1])]



        self.logoPath = '{}/QMigrate.png'.format('/'.join(ilib.__file__.split('/')[:-2]))

        self.MAPmax   = np.max(MAP)
        self.MarginalWindow = MarginalWindow


        self.CoaTraceVLINE  = None
        self.CoaValVLINE    = None
        
        self.CoaXYPlt       = None
        self.CoaYZPlt       = None
        self.CoaXZPlt       = None
        self.CoaXYPlt_VLINE = None
        self.CoaXYPlt_HLINE = None
        self.CoaYZPlt_VLINE = None
        self.CoaYZPlt_HLINE = None
        self.CoaXZPlt_VLINE = None
        self.CoaXZPlt_HLINE = None
        self.CoaArriavalTP  = None
        self.CoaArriavalTS  = None

    def CoalescenceImage(self,TimeSliceIndex):
        '''
            Takes the outputted coalescence values to plot a video over time
        '''



        TimeSlice = self.times[TimeSliceIndex]
        index = np.where(self.EVENT['DT'] == TimeSlice)[0][0]
        indexVal = self.LUT.coord2loc(np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])).astype(int)[0]
        indexCoord = np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])[0,:]

        print(indexVal)

        # ----------------  Defining the Plot Area -------------------
        fig = plt.figure(figsize=(30,15))
        fig.patch.set_facecolor('white')
        Coa_XYSlice  =  plt.subplot2grid((3, 5), (0, 0), colspan=2,rowspan=2)
        Coa_YZSlice  =  plt.subplot2grid((3, 5), (2, 0), colspan=2)
        Coa_XZSlice  =  plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        Coa_Trace    =  plt.subplot2grid((3, 5), (0, 3), colspan=2,rowspan=2)
        Coa_Logo     =  plt.subplot2grid((3, 5), (2, 2))
        Coa_CoaVal   =  plt.subplot2grid((3, 5), (2, 3), colspan=2)


        # ---------------- Plotting the Traces -----------
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        ENIn = np.where(self.times == self.EVENT['DT'].iloc[-1])[0][0]

        # ------------ Defining the stations in alphabetical order --------
        if self.RangeOrder == True: 
            ttp = self.LUT.get_value_at('TIME_P',np.array([indexVal[0],indexVal[1],indexVal[2]]))[0]
            StaInd = np.argsort(ttp)[::-1]
        else:
            StaInd = np.argsort(self.DATA.StationInformation['Name'])[::-1]


        # ------------ Defining the stations in alphabetical order --------


        for ii in range(self.DATA.signal.shape[1]): 
           if self.FilteredSignal == False:
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.signal[0,ii,:]/np.max(abs(self.DATA.signal[0,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'r',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.signal[1,ii,:]/np.max(abs(self.DATA.signal[1,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'b',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.signal[2,ii,:]/np.max(abs(self.DATA.signal[2,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'g',linewidth=0.5)
           else:
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[0,ii,:]/np.max(abs(self.DATA.FilteredSignal[0,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'r',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[1,ii,:]/np.max(abs(self.DATA.FilteredSignal[1,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'b',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[2,ii,:]/np.max(abs(self.DATA.FilteredSignal[2,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'g',linewidth=0.5)

        # ---------------- Plotting the Station Travel Times -----------
        for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal[0],indexVal[1],indexVal[2]]))[0].shape[0]):
           tp = self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0],indexVal[1],indexVal[2]]))[0][i])
           ts = self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0],indexVal[1],indexVal[2]]))[0][i])

           if i == 0:
               TP = tp
               TS = ts
           else:
               TP = np.append(TP,tp)
               TS = np.append(TS,ts)


        self.CoaArriavalTP = Coa_Trace.scatter(TP,(StaInd+1),40,'pink',marker='v')
        self.CoaArriavalTS = Coa_Trace.scatter(TS,(StaInd+1),40,'purple',marker='v')

#        Coa_Trace.set_ylim([0,ii+2])
        Coa_Trace.set_xlim([self.DATA.startTime+timedelta(seconds=1.6),np.max(TS)])
        #Coa_Trace.get_xaxis().set_ticks([])
        Coa_Trace.yaxis.tick_right()
        Coa_Trace.yaxis.set_ticks(StaInd+1)
        Coa_Trace.yaxis.set_ticklabels(self.DATA.StationInformation['Name'])
        self.CoaTraceVLINE = Coa_Trace.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])],0,1000,linestyle='--',linewidth=2,color='r')

        # ------------- Plotting the Coalescence Function ----------- 
        Coa_CoaVal.plot(self.EVENT['DT'],self.EVENT['COA'])
        Coa_CoaVal.set_ylabel('Coalescence Value')
        Coa_CoaVal.set_xlabel('Date-Time')
        Coa_CoaVal.yaxis.tick_right()
        Coa_CoaVal.yaxis.set_label_position("right")
        Coa_CoaVal.set_xlim([self.EVENT['DT'].iloc[0],self.EVENT['DT'].iloc[-1]])
        Coa_CoaVal.format_xdate = mdates.DateFormatter('%Y-%m-%d') #FIX - Not working
        for tick in Coa_CoaVal.get_xticklabels():
                tick.set_rotation(45)
        self.CoaValVLINE   = Coa_CoaVal.axvline(TimeSlice,0,1000,linestyle='--',linewidth=2,color='r')



        #  ------------- Plotting the Coalescence Value Slices -----------
        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        self.CoaXYPlt = Coa_XYSlice.pcolormesh(gridX,gridY,self.MAP[:,:,int(indexVal[2]),int(TimeSliceIndex-STIn)]/self.MAPmax,vmin=0,vmax=1,cmap=self.CMAP)
        Coa_XYSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_XYSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        self.CoaXYPlt_VLINE = Coa_XYSlice.axvline(x=indexCoord[0],linestyle='--',linewidth=2,color='k')
        self.CoaXYPlt_HLINE = Coa_XYSlice.axhline(y=indexCoord[1],linestyle='--',linewidth=2,color='k')


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2]]
        self.CoaYZPlt = Coa_YZSlice.pcolormesh(gridX,gridY,self.MAP[:,int(indexVal[1]),:,int(TimeSliceIndex-STIn)]/self.MAPmax,vmin=0,vmax=1,cmap=self.CMAP)
        Coa_YZSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_YZSlice.set_ylim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        self.CoaYZPlt_VLINE = Coa_YZSlice.axvline(x=indexCoord[0],linestyle='--',linewidth=2,color='k')
        self.CoaYZPlt_HLINE = Coa_YZSlice.axhline(y=indexCoord[2],linestyle='--',linewidth=2,color='k')
        Coa_YZSlice.invert_yaxis()


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        self.CoaXZPlt = Coa_XZSlice.pcolormesh(gridX,gridY,np.transpose(self.MAP[int(indexVal[0]),:,:,int(TimeSliceIndex-STIn)])/self.MAPmax,vmin=0,vmax=1,cmap=self.CMAP)
        Coa_XZSlice.set_xlim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_XZSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        self.CoaXZPlt_VLINE = Coa_XZSlice.axvline(x=indexCoord[2],linestyle='--',linewidth=2,color='k')
        self.CoaXZPlt_HLINE = Coa_XZSlice.axhline(y=indexCoord[1],linestyle='--',linewidth=2,color='k')



        #  ------------- Plotting the station Locations -----------
        Coa_XYSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Latitude'],15,'k',marker='^')
        Coa_YZSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Elevation'],15,'k',marker='^')
        Coa_XZSlice.scatter(self.LUT.station_data['Elevation'],self.LUT.station_data['Latitude'],15,'k',marker='<')
        for i,txt in enumerate(self.LUT.station_data['Name']):
            Coa_XYSlice.annotate(txt,[self.LUT.station_data['Longitude'][i],self.LUT.station_data['Latitude'][i]])


    #  ------------- Plotting the XYFiles -----------
        if self.XYFiles != None:
           XYFiles = pd.read_csv(self.XYFiles,names=['File','Color','Linewidth','Linestyle'])
           c=0
           for ff in XYFiles['File']:
                XYF = pd.read_csv(ff,names=['X','Y'])       
                Coa_XYSlice.plot(XYF['X'],XYF['Y'],linestyle=XYFiles['Linestyle'].iloc[c],linewidth=XYFiles['Linewidth'].iloc[c],color=XYFiles['Color'].iloc[c])
                c+=1



        #  ------------- Plotting the station Locations -----------
        try:
            Coa_Logo.axis('off')
            im = mpimg.imread(self.logoPath)
            Coa_Logo.imshow(im)
            Coa_Logo.text(150, 200, r'CoalescenceVideo', fontsize=14,style='italic')
        except:
            'Logo not plotting'

        return fig


    def _CoalescenceVideo_update(self,frame):
        frame = int(frame)
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        TimeSlice  = self.times[int(frame)]
        index      = np.where(self.EVENT['DT'] == TimeSlice)[0][0]
        indexVal = self.LUT.coord2loc(np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])).astype(int)[0]
        indexCoord = np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])[0,:]

        # Updating the Coalescence Value and Trace Lines
        self.CoaTraceVLINE.set_xdata(TimeSlice)
        self.CoaValVLINE.set_xdata(TimeSlice)

        # Updating the Coalescence Maps
        self.CoaXYPlt.set_array((self.MAP[:,:,indexVal[2],int(STIn-frame)]/self.MAPmax)[:-1,:-1].ravel())
        self.CoaYZPlt.set_array((self.MAP[:,indexVal[1],:,int(STIn-frame)]/self.MAPmax)[:-1,:-1].ravel())
        self.CoaXZPlt.set_array((np.transpose(self.MAP[indexVal[0],:,:,int(STIn-frame)])/self.MAPmax)[:-1,:-1].ravel())

        # Updating the Coalescence Lines
        self.CoaXYPlt_VLINE.set_xdata(indexCoord[0])
        self.CoaXYPlt_HLINE.set_ydata(indexCoord[1])
        self.CoaYZPlt_VLINE.set_xdata(indexCoord[0])
        self.CoaYZPlt_HLINE.set_ydata(indexCoord[2])
        self.CoaXZPlt_VLINE.set_xdata(indexCoord[2])
        self.CoaXZPlt_HLINE.set_ydata(indexCoord[1])


        # Updating the station travel-times
        for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0].shape[0]):
            try:
                tp = np.argmin(abs((self.times.astype(datetime) - (TimeSlice.astype(datetime) + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])))/timedelta(seconds=1)))
                ts = np.argmin(abs((self.times.astype(datetime) - (TimeSlice.astype(datetime) + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])))/timedelta(seconds=1)))
            except:
                tp=0
                ts=0



            if i == 0:
                TP = tp
                TS = ts
            else:
                TP = np.append(TP,tp)
                TS = np.append(TS,ts)

        self.CoaArriavalTP.set_offsets(np.c_[TP,(np.arange(self.DATA.signal.shape[1])+1)])
        self.CoaArriavalTS.set_offsets(np.c_[TS,(np.arange(self.DATA.signal.shape[1])+1)])


        # # Updating the station travel-times
        # self.CoaArriavalTP.remove()
        # self.CoaArriavalTS.remove()
        # self.CoaArriavalTS = None
        # self.CoaArriavalTP = None
        # for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0].shape[0]):
        #     if i == 0:
        #         TP = TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])
        #         TS = TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])
        #     else:
        #         TP = np.append(TP,(TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])))
        #         TS = np.append(TS,(TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])))

        # self.CoaArriavalTP = Coa_Trace.scatter(TP,(np.arange(self.DATA.signal.shape[1])+1),15,'r',marker='v')
        # self.CoaArriavalTS = Coa_Trace.scatter(TS,(np.arange(self.DATA.signal.shape[1])+1),15,'b',marker='v')


    def CoalescenceTrace(self,SaveFilename=None):

        # Determining the maginal window value from the coalescence function
        mMAP = self.CoaMAP
        # mMAP = np.log(np.sum(np.exp(mMAP),axis=-1))
        # mMAP = mMAP/np.max(mMAP)
        # mMAP_Cutoff = np.percentile(mMAP,95)
        # mMAP[mMAP < mMAP_Cutoff] = mMAP_Cutoff 
        # mMAP = mMAP - mMAP_Cutoff 
        # mMAP = mMAP/np.max(mMAP)
        indexVal = np.where(mMAP == np.max(mMAP))
        indexCoord = self.LUT.xyz2coord(self.LUT.loc2xyz(np.array([[indexVal[0][0],indexVal[1][0],indexVal[2][0]]])))




        # Looping through all stations
        ii=0
        while ii < self.DATA.signal.shape[1]: 
                fig = plt.figure(figsize=(30,15))

                # Defining the plot
                fig.patch.set_facecolor('white')
                XTrace_Seis  =  plt.subplot(322)
                YTrace_Seis  =  plt.subplot(324)
                ZTrace_Seis  =  plt.subplot(321)
                P_Onset      =  plt.subplot(323)
                S_Onset      =  plt.subplot(326)



                # --- If trace is blank then remove and don't plot ---



                # Plotting the X-trace
                XTrace_Seis.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[0,ii,:]/np.max(abs(self.DATA.FilteredSignal[0,ii,:])))*self.TraceScaling,'r',linewidth=0.5)
                YTrace_Seis.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[1,ii,:]/np.max(abs(self.DATA.FilteredSignal[1,ii,:])))*self.TraceScaling,'b',linewidth=0.5)
                ZTrace_Seis.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[2,ii,:]/np.max(abs(self.DATA.FilteredSignal[2,ii,:])))*self.TraceScaling,'g',linewidth=0.5)
                P_Onset.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),self.DATA.SNR_P[ii,:],'r',linewidth=0.5)
                S_Onset.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),self.DATA.SNR_S[ii,:],'b',linewidth=0.5)


                # Defining Pick and Error
                PICKS_df = self.StationPick['Pick']
                STATION_pick = PICKS_df[PICKS_df['Name'] == self.LUT.station_data['Name'][ii]].reset_index(drop=True)
                if len(STATION_pick) > 0:
                    STATION_pick = STATION_pick.replace('-1.0',np.nan)


                    for jj in range(len(STATION_pick)):
                        if np.isnan(STATION_pick['PickError'].iloc[jj]):
                            continue 

                        if STATION_pick['Phase'].iloc[jj] == 'P':
                            ZTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            ZTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            ZTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                            # S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            # S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            # S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                            yy = gaussian_func(self.StationPick['GAU_P'][ii]['xdata'],self.StationPick['GAU_P'][ii]['popt'][0],self.StationPick['GAU_P'][ii]['popt'][1],self.StationPick['GAU_P'][ii]['popt'][2])
                            P_Onset.plot(self.StationPick['GAU_P'][ii]['xdata_dt'],yy)
                            P_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            P_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            P_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))




                        else:
                            YTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            YTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            YTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                            XTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            XTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            XTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                            yy = gaussian_func(self.StationPick['GAU_S'][ii]['xdata'],self.StationPick['GAU_S'][ii]['popt'][0],self.StationPick['GAU_S'][ii]['popt'][1],self.StationPick['GAU_S'][ii]['popt'][2])
                            S_Onset.plot(self.StationPick['GAU_S'][ii]['xdata_dt'],yy)

                            S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                ZTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] ),color='red')
                ZTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.MarginalWindow - 0.1*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
                ZTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.MarginalWindow + 0.1*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
                P_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
                P_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.MarginalWindow - 0.1*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
                P_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.MarginalWindow + 0.1*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')


                YTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
                YTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.MarginalWindow - 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
                YTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.MarginalWindow + 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')

                XTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
                XTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.MarginalWindow - 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red')
                XTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.MarginalWindow + 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red')


                S_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
                S_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] - self.MarginalWindow - 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')
                S_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=(self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii] + self.MarginalWindow + 0.1*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii])),color='red',linestyle='--')



                P_Onset.axhline(self.StationPick['GAU_P'][ii]['PickThreshold'])
                S_Onset.axhline(self.StationPick['GAU_S'][ii]['PickThreshold'])

                # Refining the window as around the pick time
                MINT = pd.to_datetime(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=0.5*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]))
                MAXT = pd.to_datetime(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=1.5*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]))

                XTrace_Seis.set_xlim([MINT,MAXT])
                YTrace_Seis.set_xlim([MINT,MAXT])
                ZTrace_Seis.set_xlim([MINT,MAXT])
                P_Onset.set_xlim([MINT,MAXT])
                S_Onset.set_xlim([MINT,MAXT])

                P_Onset.set_ylim([0.0, P_Onset.get_ylim()[1]])
                S_Onset.set_ylim([0.0, S_Onset.get_ylim()[1]])


                fig.suptitle('Trace for Station {} - PPick = {}, SPick = {}'.format(self.LUT.station_data['Name'][ii],self.StationPick['GAU_P'][ii]['PickValue'],self.StationPick['GAU_S'][ii]['PickValue']))


                
                if SaveFilename == None:
                   plt.show()
                else:
                   plt.savefig('{}_CoalescenceTrace_{}.pdf'.format(SaveFilename,self.LUT.station_data['Name'][ii]))
                   plt.close("all")


                ii+=1
            

    def CoalescenceVideo(self,SaveFilename=None):
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        ENIn = np.where(self.times == self.EVENT['DT'].iloc[-1])[0][0]

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=4, metadata=dict(artist='Ulvetanna'), bitrate=1800)


        FIG = self.CoalescenceImage(STIn)
        ani = animation.FuncAnimation(FIG, self._CoalescenceVideo_update, frames=np.linspace(STIn,ENIn-1,200),blit=False,repeat=False) 

        if SaveFilename == None:
            plt.show()
        else:
            ani.save('{}_CoalescenceVideo.mp4'.format(SaveFilename),writer=writer)

    def CoalescenceMarginal(self,SaveFilename=None, Earthquake=None):
        '''
            Generates a Marginal window about the event to determine the error.

            # Redefine the marginal as instead of the whole coalescence period, gaussian fit to the coalescence value 
            then take the 1st std to define the time window and use this

        '''

        TimeSliceIndex = np.where(self.times == self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])])[0][0]
        TimeSlice = self.times[TimeSliceIndex]
        index = np.where(self.EVENT['DT'] == TimeSlice)[0][0]
        indexVal1 = self.LUT.coord2loc(np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])).astype(int)[0]
        indexCoord = np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])[0,:]

        # Determining the maginal window value from the coalescence function
        mMAP = self.CoaMAP
        indexVal = np.where(mMAP == np.max(mMAP))
        indexCoord = self.LUT.xyz2coord(self.LUT.loc2xyz(np.array([[indexVal[0][0],indexVal[1][0],indexVal[2][0]]])))


        # Defining the plots to be represented
        fig = plt.figure(figsize=(30,15))
        fig.patch.set_facecolor('white')
        Coa_XYSlice  =  plt.subplot2grid((3, 5), (0, 0), colspan=2,rowspan=2)
        Coa_YZSlice  =  plt.subplot2grid((3, 5), (2, 0), colspan=2)
        Coa_XZSlice  =  plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        Coa_Trace    =  plt.subplot2grid((3, 5), (0, 3), colspan=2,rowspan=2)
        Coa_Logo     =  plt.subplot2grid((3, 5), (2, 2))
        Coa_CoaVal   =  plt.subplot2grid((3, 5), (2, 3), colspan=2)



        # ---------------- Plotting the Traces -----------
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        ENIn = np.where(self.times == self.EVENT['DT'].iloc[-1])[0][0]
        #print(STIn,ENIn)



        # --------------- Ordering by distance to event --------------
        if self.RangeOrder == True: 
            ttp = self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0]
            StaInd = abs(np.argsort(np.argsort(ttp))-np.max(np.argsort(np.argsort(ttp))))
        else:
            StaInd = np.argsort(self.DATA.StationInformation['Name'])[::-1]

        # ---------------- Determining Station Travel-Times -----------
        for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0].shape[0]):
           tp = self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][i])
           ts = self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][i])

           if i == 0:
               TP = tp
               TS = ts
           else:
               TP = np.append(TP,tp)
               TS = np.append(TS,ts)


        DDD=np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate))
        # print(pd.to_datetime(pd.Series(DDD)))
        # DDD_min = np.argmin(abs(pd.to_datetime(pd.Series(DDD)) - (self.DATA.startTime+timedelta(seconds=1.6))))
        # DDD_max = np.argmin(abs(pd.to_datetime(pd.Series(DDD)) - (np.max(TS))))

        for ii in range(self.DATA.signal.shape[1]): 
           if self.FilteredSignal == False:
                   Coa_Trace.plot(DDD,(self.DATA.signal[0,ii,:]/np.max(abs(self.DATA.signal[0,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'r',linewidth=0.5,zorder=1)
                   Coa_Trace.plot(DDD,(self.DATA.signal[1,ii,:]/np.max(abs(self.DATA.signal[1,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'b',linewidth=0.5,zorder=2)
                   Coa_Trace.plot(DDD,(self.DATA.signal[2,ii,:]/np.max(abs(self.DATA.signal[2,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'g',linewidth=0.5,zorder=3)
           else:
                   Coa_Trace.plot(DDD,(self.DATA.FilteredSignal[0,ii,:]/np.max(abs(self.DATA.FilteredSignal[0,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'r',linewidth=0.5,zorder=1)
                   Coa_Trace.plot(DDD,(self.DATA.FilteredSignal[1,ii,:]/np.max(abs(self.DATA.FilteredSignal[1,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'b',linewidth=0.5,zorder=2)
                   Coa_Trace.plot(DDD,(self.DATA.FilteredSignal[2,ii,:]/np.max(abs(self.DATA.FilteredSignal[2,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'g',linewidth=0.5,zorder=3)

        # ---------------- Plotting Station Travel-Times -----------

        self.CoaArriavalTP = Coa_Trace.scatter(TP,(StaInd+1),50,'pink',marker='v',zorder=4,linewidth=0.1,edgecolors='black')
        self.CoaArriavalTS = Coa_Trace.scatter(TS,(StaInd+1),50,'purple',marker='v',zorder=5,linewidth=0.1,edgecolors='black')

#        Coa_Trace.set_ylim([0,ii+2])
        Coa_Trace.set_xlim([self.DATA.startTime+timedelta(seconds=1.6),np.max(TS)])
        #Coa_Trace.get_xaxis().set_ticks([])
        Coa_Trace.yaxis.tick_right()
        Coa_Trace.yaxis.set_ticks(StaInd+1)
        Coa_Trace.yaxis.set_ticklabels(self.DATA.StationInformation['Name'])
        self.CoaTraceVLINE = Coa_Trace.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])],0,1000,linestyle='--',linewidth=2,color='r')

        # ------------- Plotting the Coalescence Function ----------- 
        Coa_CoaVal.plot(self.EVENT['DT'],self.EVENT['COA'])
        Coa_CoaVal.set_ylabel('Coalescence Value')
        Coa_CoaVal.set_xlabel('Date-Time')
        Coa_CoaVal.yaxis.tick_right()
        Coa_CoaVal.yaxis.set_label_position("right")
        Coa_CoaVal.set_xlim([self.EVENT['DT'].iloc[0],self.EVENT['DT'].iloc[-1]])
        Coa_CoaVal.format_xdate = mdates.DateFormatter('%Y-%m-%d') #FIX - Not working
        for tick in Coa_CoaVal.get_xticklabels():
                tick.set_rotation(45)
        self.CoaValVLINE   = Coa_CoaVal.axvline(TimeSlice,0,1000,linestyle='--',linewidth=2,color='r')

        # -------------- Determining Error ellipse for Covariance -----------
        dCo = abs(indexCoord - self.LUT.xyz2coord(self.LUT.loc2xyz(np.array([[indexVal[0][0] + int(Earthquake['Covariance_ErrX'].iloc[0]/self.LUT.cell_size[0]),indexVal[1][0] + int(Earthquake['Covariance_ErrY'].iloc[0]/self.LUT.cell_size[1]),indexVal[2][0] + int(Earthquake['Covariance_ErrZ'].iloc[0]/self.LUT.cell_size[2])]]))))

        ellipse_XY = patches.Ellipse((Earthquake['Covariance_X'].iloc[0], Earthquake['Covariance_Y'].iloc[0]), 2*dCo[0][0], 2*dCo[0][1], angle=0, linewidth=2, fill=False)
        ellipse_YZ = patches.Ellipse((Earthquake['Covariance_X'].iloc[0], Earthquake['Covariance_Z'].iloc[0]), 2*dCo[0][0], 2*dCo[0][2], angle=0, linewidth=2, fill=False)
        ellipse_XZ = patches.Ellipse((Earthquake['Covariance_Z'].iloc[0], Earthquake['Covariance_Y'].iloc[0]), 2*dCo[0][2], 2*dCo[0][1], angle=0, linewidth=2, fill=False)



        # ------------- Spatial Function  ----------- 

        # Plotting the marginal window
        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        rect = Rectangle((np.min(gridX), np.min(gridY)), np.max(gridX)-np.min(gridX),np.max(gridY)-np.min(gridY))
        pc = PatchCollection([rect], facecolor='k')
        Coa_XYSlice.add_collection(pc)
        Coa_XYSlice.pcolormesh(gridX,gridY,mMAP[:,:,int(indexVal[2][0])],cmap=self.CMAP,edgecolors='face')
        #CS = Coa_XYSlice.contour(gridX,gridY,mMAP[:,:,int(indexVal[2][0])],levels=[0.65,0.75,0.95],colors=('g','m','k'))
        #Coa_XYSlice.clabel(CS, inline=1, fontsize=10)
        Coa_XYSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_XYSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        Coa_XYSlice.axvline(x=indexCoord[0][0],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XYSlice.axhline(y=indexCoord[0][1],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XYSlice.scatter(Earthquake['Gaussian_X'].iloc[0],Earthquake['Gaussian_Y'].iloc[0],150,c='pink',marker='*')
        Coa_XYSlice.scatter(Earthquake['Covariance_X'].iloc[0],Earthquake['Covariance_Y'].iloc[0],150,c='blue',marker='*')
        Coa_XYSlice.add_patch(ellipse_XY)


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2]]
        rect = Rectangle((np.min(gridX), np.min(gridY)), np.max(gridX)-np.min(gridX),np.max(gridY)-np.min(gridY))
        pc = PatchCollection([rect], facecolor='k')
        Coa_YZSlice.add_collection(pc)
        Coa_YZSlice.pcolormesh(gridX,gridY,mMAP[:,int(indexVal[1][0]),:],cmap=self.CMAP,edgecolors='face')
        #CS = Coa_YZSlice.contour(gridX,gridY,mMAP[:,int(indexVal[1][0]),:], levels=[0.65,0.75,0.95],colors=('g','m','k'))
        #Coa_YZSlice.clabel(CS, inline=1, fontsize=10)
        Coa_YZSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_YZSlice.set_ylim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_YZSlice.axvline(x=indexCoord[0][0],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_YZSlice.axhline(y=indexCoord[0][2],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_YZSlice.scatter(Earthquake['Gaussian_X'].iloc[0],Earthquake['Gaussian_Z'].iloc[0],150,c='pink',marker='*')
        Coa_YZSlice.scatter(Earthquake['Covariance_X'].iloc[0],Earthquake['Covariance_Z'].iloc[0],150,c='blue',marker='*')
        Coa_YZSlice.add_patch(ellipse_YZ)
        Coa_YZSlice.invert_yaxis()


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        rect = Rectangle((np.min(gridX), np.min(gridY)), np.max(gridX)-np.min(gridX),np.max(gridY)-np.min(gridY))
        pc = PatchCollection([rect], facecolor='k')
        Coa_XZSlice.add_collection(pc)
        Coa_XZSlice.pcolormesh(gridX,gridY,mMAP[int(indexVal[0][0]),:,:].transpose(),cmap=self.CMAP,edgecolors='face')
        #CS = Coa_XZSlice.contour(gridX,gridY,mMAP[int(indexVal[0][0]),:,:].transpose(),levels =[0.65,0.75,0.95],colors=('g','m','k'))
        Coa_XZSlice.set_xlim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_XZSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        Coa_XZSlice.axvline(x=indexCoord[0][2],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XZSlice.axhline(y=indexCoord[0][1],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XZSlice.scatter(Earthquake['Gaussian_Z'].iloc[0],Earthquake['Gaussian_Y'].iloc[0],150,c='pink',marker='*')
        Coa_XZSlice.scatter(Earthquake['Covariance_Z'].iloc[0],Earthquake['Covariance_Y'].iloc[0],150,c='blue',marker='*')
        Coa_XZSlice.add_patch(ellipse_XZ)

        # Plotting the station locations
        Coa_XYSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Latitude'],15,marker='^',color=self.LineStationColor)
        Coa_YZSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Elevation'],15,marker='^',color=self.LineStationColor)
        Coa_XZSlice.scatter(self.LUT.station_data['Elevation'],self.LUT.station_data['Latitude'],15,marker='<',color=self.LineStationColor)
        for i,txt in enumerate(self.LUT.station_data['Name']):
            Coa_XYSlice.annotate(txt,[self.LUT.station_data['Longitude'][i],self.LUT.station_data['Latitude'][i]],color=self.LineStationColor)

        # Plotting the XYFiles
        if self.XYFiles != None:
           XYFiles = pd.read_csv(self.XYFiles,names=['File','Color','Linewidth','Linestyle'])
           c=0
           for ff in XYFiles['File']:
                XYF = pd.read_csv(ff,names=['X','Y'])       
                Coa_XYSlice.plot(XYF['X'],XYF['Y'],linestyle=XYFiles['Linestyle'].iloc[c],linewidth=XYFiles['Linewidth'].iloc[c],color=XYFiles['Color'].iloc[c])
                c+=1

        # Plotting the logo
        try:
            Coa_Logo.axis('off')
            im = mpimg.imread(self.logoPath)
            Coa_Logo.imshow(im)
            Coa_Logo.text(150, 200, r'Earthquake Location Error', fontsize=10,style='italic')
        except:
            'Logo not plotting'

        if SaveFilename == None:
            plt.show()

        else:
            plt.savefig('{}_EventLocationError.pdf'.format(SaveFilename),dpi=400)
            plt.close('all')




class SeisScanParam:
    '''
       Class that reads in a user defined parameter file for all the required
    scanning Information. Currently takes the 

      _set_param - Definition of the path for the Parameter file to be read

    '''

    def __init__(self, param = None):
        self.lookup_table = None
        self.seis_reader = None
        self.bp_filter_p1 = [2.0, 16.0, 3]
        self.bp_filter_s1 = [2.0, 12.0, 3]
        self.onset_win_p1 = [0.2, 1.0]
        self.onset_win_s1 = [0.2, 1.0]
        self.station_p1 = None
        self.station_s1 = None
        self.detection_threshold = 3.0
        self.detection_downsample = 5
        self.detection_window = 3.0
        self.minimum_velocity = 3000.0
        self.marginal_window = [0.5, 3000.0]
        self.location_method = "Mean"
        self.time_step = 10
        self.StartDateTime=None
        self.EndDateTime=None
        self.Decimate=[1,1,1]

        if param:
            self.load(param)

    def _set_param(self, param):

        # Defining the Model Types to load LUT from
        type = _find(param,("MODEL","Type"))
        
        if (type == "MATLAB"):
            path = _find(param,("MODEL","Path"))
            if path:
                decimate = _find(param,("MODEL","Decimate"))
                self.lookup_table = mload.lut02(path, decimate)

        if (type == "QMigrate"):
            path = _find(param,("MODEL","Path"))
            if path:
                decimate = _find(param,("MODEL","Decimate"))
                self.lookup_table = mload.lut02(path, decimate)

        if (type == "NLLoc"):
            path = _find(param,("MODEL","Path"))
            if path:
                decimate = _find(param,("MODEL","Decimate"))
                self.lookup_table = mload.lut02(path, decimate)




        # Defining the Seimsic Data to load the information from
        type = _find(param,("SEISMIC","Type"))
        if (type == 'MSEED'):
            path = _find(param,("SEISMIC","Path"))
            if path:
                self.seis_reader = mload.mseed_reader(path)

        # ~ Other possible types to add DAT,SEGY,RAW


        # Defining the Time-Period to scan across 
        scn = _find(param,"SCAN")
        if scn:
            self.StartDateTime = _find(scn,"Start_DateTime",self.StartDateTime)
            self.EndDateTime   = _find(scn,"End_DateTime",self.EndDateTime)
            self.StartDateTime = datetime.strptime(self.StartDateTime,'%Y-%m-%dT%H:%M:%S.%f')
            self.EndDateTime   = datetime.strptime(self.EndDateTime,'%Y-%m-%dT%H:%M:%S.%f')

        # Defining the Parameters for the Coalescence
        scn = _find(param,("PARAM"))
        if scn:
            self.time_step            = _find(scn,"TimeStep",self.time_step)
            self.station_p1           = _find(scn,"StationSelectP",self.station_p1)
            self.station_s1           = _find(scn,"StationSelectS",self.station_s1)
            self.bp_filter_p1         = _find(scn,"SigFiltP1Hz",self.bp_filter_p1)
            self.bp_filter_s1         = _find(scn,"SigFiltS1Hz",self.bp_filter_s1)
            self.onset_win_p1         = _find(scn,"OnsetWinP1Sec",self.onset_win_p1)
            self.onset_win_s1         = _find(scn,"OnsetWinS1Sec",self.onset_win_s1)
            self.detection_downsample = _find(scn,"DetectionDownsample",self.detection_downsample)
            self.detection_window     = _find(scn,"DetectionWindow",self.detection_window)
            self.minimum_velocity     = _find(scn,"MinimumVelocity",self.minimum_velocity)
            self.marginal_window      = _find(scn,"MarginalWindow",self.marginal_window)
            self.location_method      = _find(scn,"LocationMethod",self.location_method)

    def _load_json(self, json_file):
        param = None
        with open(json_file,'r') as fp:
            param = json.load(fp)
        return param

    def load(self, file):
        param = self._load_json(file)
        self._set_param(param)


class SeisScan:

    def __init__(self, DATA, LUT, reader=None, param=None, output_path=None, output_name=None):
        
        lut = cmod.LUT()
        lut.load(LUT)
        self.sample_rate = 1000.0
        self.seis_reader = None
        self.lookup_table = lut
        self.DATA = DATA 

        if param is None:
            param = SeisScanParam()


        self.keep_map = False

        ttmax = np.max(lut.fetch_map('TIME_S'))
        self.pre_pad   = None
        self.post_pad  = round(ttmax + ttmax*0.05)
        self.time_step = 10.0

        self.daten = None
        self.dsnr  = None
        self.dloc  = None

        self.PickThreshold = 1.0
        

        self.bp_filter_p1 = param.bp_filter_p1
        self.bp_filter_s1 = param.bp_filter_s1
        self.onset_win_p1 = param.onset_win_p1
        self.onset_win_s1 = param.onset_win_s1
        self.boxcar_p1 = 0.050
        self.boxcar_s1 = 0.100
        self.station_p1 = param.station_p1
        self.station_s1 = param.station_s1
        self.detection_threshold = param.detection_threshold

        if output_path is not None:
            self.output = SeisOutFile(output_path, output_name)

        else:
            self.outputps = None

        self.raw_data = dict()
        self.filt_data = dict()
        self.onset_data = dict()

        self._initialized = False
        self._station_name = None
        self._station_p1_flg = None
        self._station_s1_flg = None
        self._station_file = None
        self._map = None
        self._daten = None
        self._dsnr = None
        self.snr = None 
        self._data = None
        self._onset_centred = False
        self._coalescenceMseed = None
        self._DeepLearning = False


        self.NumberOfCores = 1

        self.DetectionThreshold = 1
        self.MarginalWindow     = 30
        self.MinimumRepeat      = 30
        self.PercentageTT       = 0.1
        self.CoalescenceGrid    = False
        self.CoalescenceVideo   = False
        self.CoalescencePicture = False
        self.CoalescenceTrace   = False
        self.CutMSEED           = False
        self.PickingType        = 'Gaussian'
        self.LocationError      = 0.95
        self.NormaliseCoalescence = False
        self.Output_SampleRate = None 


        #self.plot = SeisPlot(lut)

        self.MAP = None
        self.CoaMAP = None
        self.EVENT = None
        self.XYFiles = None


        # ---- Printing to the terminal the initial information -----
        print('==============================================================================================================================')
        print('==============================================================================================================================')
        print('   QMigrate - Coalescence Scanning : PATH:{} - NAME:{}'.format(self.output.path, self.output.name))
        print('==============================================================================================================================')
        print('==============================================================================================================================')
        print('')


    def _pre_proc_p1(self, sig_z, srate):
        lc, hc, ord = self.bp_filter_p1           # Apply - Bandpass Filter  - information defined in ParameterFile/Inputs
        sig_z = filter(sig_z, srate, lc, hc, ord) # Apply - Butter filter
        self.DATA.FilteredSignal[2,:,:] = sig_z
        return sig_z

    def _pre_proc_s1(self, sig_e, sig_n, srate):
        lc, hc, ord = self.bp_filter_s1               # Apply - Bandpass Filter  - information defined in ParameterFile/Inputs
        sig_e = filter(sig_e, srate, lc, hc, ord) # Apply - Butter filter on E
        sig_n = filter(sig_n, srate, lc, hc, ord) # Apply - Butter filter on N
        self.DATA.FilteredSignal[0,:,:] = sig_n
        self.DATA.FilteredSignal[1,:,:] = sig_e
        return sig_e, sig_n

    def _compute_onset_p1(self, sig_z, srate):
        stw, ltw = self.onset_win_p1             # Define the STW and LTW for the onset function
        stw = int(stw * srate) + 1               # Changes the onset window to actual samples
        ltw = int(ltw * srate) + 1               # Changes the onset window to actual samples
        sig_z = self._pre_proc_p1(sig_z, srate)  # Apply the pre-processing defintion
        self.filt_data['sigz'] = sig_z           # defining the data to pass 
        sig_z_raw,sig_z = onset(sig_z, stw, ltw,centred=self._onset_centred)           # Determine the onset function using definition
        self.onset_data['sigz'] = sig_z          # Define the onset function from the data 
        return sig_z_raw,sig_z

    def _compute_onset_s1(self, sig_e, sig_n, srate):
        stw, ltw = self.onset_win_s1                                            # Define the STW and LTW for the onset function
        stw = int(stw * srate) + 1                                              # Changes the onset window to actual samples
        ltw = int(ltw * srate) + 1                                              # Changes the onset window to actual samples
        sig_e, sig_n = self._pre_proc_s1(sig_e, sig_n, srate)                   # Apply the pre-processing defintion
        self.filt_data['sige'] = sig_e                                          # Defining filtered signal to pass
        self.filt_data['sign'] = sig_n                                          # Defining filtered signal to pass
        sig_e_raw,sig_e = onset(sig_e, stw, ltw,centred=self._onset_centred)                                          # Determine the onset function from the filtered signal
        sig_n_raw,sig_n = onset(sig_n, stw, ltw,centred=self._onset_centred)                                          # Determine the onset function from the filtered signal
        self.onset_data['sige'] = sig_e                                         # Define the onset function from the data
        self.onset_data['sign'] = sig_n                                         # Define the onset function from the data                
        snr = np.sqrt((sig_e * sig_e + sig_n * sig_n)/2.)
        snr_raw = np.sqrt((sig_e_raw * sig_e_raw + sig_n_raw * sig_n_raw)/2.)                            # Define the combined onset function from E & N
        self.onset_data['sigs'] = snr
        return snr_raw,snr


    def _compute(self, cstart,cend, samples,station_avaliability):

        srate = self.sample_rate


        avaInd = np.where(station_avaliability == 1)[0]
        sige = samples[0]
        sign = samples[1]
        sigz = samples[2]

        # Demeaning the data 
        #sige = sige - np.mean(sige,axis=1)
        #sign = sign - np.mean(sign,axis=1)
        #sigz = sigz - np.mean(sigz,axis=1)
        if self._DeepLearning == False:
            snr_p1_raw,snr_p1 = self._compute_onset_p1(sigz, srate)
            snr_s1_raw,snr_s1 = self._compute_onset_s1(sige, sign, srate)
            self.DATA.SNR_P = snr_p1
            self.DATA.SNR_S = snr_s1
            self.DATA.SNR_P_raw = snr_p1_raw
            self.DATA.SNR_S_raw = snr_s1_raw

        else:

            print('Deep Learning Coalescence currently under development - Please come back soon ...')
            # print('Applying Machine Learning Technqiue')
            # DL = DeepLearningPhaseDetection(sign,sige,sigz,srate)  
            # self.DATA.SNR_P     = DL.prob_P
            # self.DATA.SNR_S     = DL.prob_S
            # self.DATA.SNR_P_raw = DL.prob_P
            # self.DATA.SNR_S_raw = DL.prob_S

        #self._Gaussian_Coalescence()


        snr = np.concatenate((self.DATA.SNR_P, self.DATA.SNR_S))
        snr[np.isnan(snr)] = 0
        
        
        ttp = self.lookup_table.fetch_index('TIME_P', srate)
        tts = self.lookup_table.fetch_index('TIME_S', srate)
        tt = np.c_[ttp, tts]
        del ttp,tts

        nchan, tsamp = snr.shape

        pre_smp = int(round(self.pre_pad * int(srate)))
        pos_smp = int(round(self.post_pad * int(srate)))
        nsamp = tsamp - pre_smp - pos_smp
        daten = 0.0 - pre_smp / srate

        ncell = tuple(self.lookup_table.cell_count)

        _map = np.zeros(ncell + (nsamp,), dtype=np.float64)

        dind      = np.zeros(nsamp, np.int64)
        dsnr      = np.zeros(nsamp, np.double)
        dsnr_norm = np.zeros(nsamp, np.double)

        # Determining the maximum coalescence through time
        ilib.scan(snr, tt, pre_smp, pos_smp, nsamp, _map, self.NumberOfCores)
        ilib.detect(_map, dsnr, dind, 0,nsamp, self.NumberOfCores)
        daten = np.arange((cstart+timedelta(seconds=self.pre_pad)), (cend + timedelta(seconds=-self.post_pad) + timedelta(seconds=1/srate)),timedelta(seconds=1/srate))
        dsnr  = np.exp((dsnr / (len(avaInd)*2)) - 1.0)
        dloc  = self.lookup_table.index2xyz(dind)

        # Determining the normalised coalescence through time 
        SumCoa = np.sum(_map,axis=(0,1,2))
        _map =  _map / SumCoa[np.newaxis,np.newaxis,np.newaxis,:]
        ilib.detect(_map, dsnr_norm, dind, 0,nsamp, self.NumberOfCores)
        dsnr_norm  = dsnr_norm * _map.shape[0] * _map.shape[1] * _map.shape[2]

        # Resetting the map as the original coalescence value
        _map =  _map * SumCoa[np.newaxis,np.newaxis,np.newaxis,:]        

        return daten, dsnr,dsnr_norm, dloc, _map


    def _continious_compute(self,starttime,endtime):
        ''' 
            Continious seismic compute from 

        '''

        # 1. variables check
        # 2. Defining the pre- and post- padding
        # 3.  

        CoaV = 1.0

        self.StartDateTime = datetime.strptime(starttime,'%Y-%m-%dT%H:%M:%S.%f')
        self.EndDateTime   = datetime.strptime(endtime,'%Y-%m-%dT%H:%M:%S.%f')

        # Setting the Coalescence MSEED to blank
        self._coalescenceMseed = None

        
        # ------- Continious Seismic Detection ------
        print('==============================================================================================================================')
        print('   DETECT - Continious Seismic Processing for {} to {}'.format(datetime.strftime(self.StartDateTime,'%Y-%m-%dT%H:%M:%S.%f'),datetime.strftime(self.EndDateTime,'%Y-%m-%dT%H:%M:%S.%f')))
        print('==============================================================================================================================')
        print('')
        print('   Parameters Specfied:')
        print('         Start Time              = {}'.format(datetime.strftime(self.StartDateTime,'%Y-%m-%dT%H:%M:%S.%f')))
        print('         End   Time              = {}'.format(datetime.strftime(self.EndDateTime,'%Y-%m-%dT%H:%M:%S.%f')))
        print('         TimeStep (s)            = {}'.format(self.time_step))
        print('         Number of CPUs          = {}'.format(self.NumberOfCores))
        print('')
        print('         Sample Rate             = {}'.format(int(self.sample_rate)))
        print('         Grid Decimation [X,Y,Z] = [{},{},{}]'.format(self.Decimate[0],self.Decimate[1],self.Decimate[2]))
        print('         Bandpass Filter P       = [{},{},{}]'.format(self.bp_filter_p1[0],self.bp_filter_p1[1],self.bp_filter_p1[2]))
        print('         Bandpass Filter S       = [{},{},{}]'.format(self.bp_filter_s1[0],self.bp_filter_s1[1],self.bp_filter_s1[2]))
        print('         Onset P [STA,LTA]       = [{},{}]'.format(self.onset_win_p1[0],self.onset_win_p1[1]))
        print('         Onset S [STA,LTA]       = [{},{}]'.format(self.onset_win_s1[0],self.onset_win_s1[1]))
        print('')
        print('==============================================================================================================================')

        # adding pre- and post-pad to remove affect from taper
        timeLen       = self.pre_pad + self.post_pad + self.time_step
        self.pre_pad  = self.pre_pad + round(timeLen*0.06)
        self.post_pad = self.post_pad + round(timeLen*0.06)

        i = 0 
        while self.EndDateTime >= (self.StartDateTime + timedelta(seconds=self.time_step*(i+1))):
            cstart =  self.StartDateTime + timedelta(seconds=self.time_step*i) + timedelta(seconds=-self.pre_pad)
            cend   =  self.StartDateTime + timedelta(seconds=self.time_step*(i+1)) + timedelta(seconds=self.post_pad)

            print('~~~~~~~~~~~~~ Processing - {} to {} ~~~~~~~~~~~~~'.format(datetime.strftime(cstart,'%Y-%m-%dT%H:%M:%S.%f'),datetime.strftime(cend,'%Y-%m-%dT%H:%M:%S.%f'))) 

            self.DATA.read_mseed(datetime.strftime(cstart,'%Y-%m-%dT%H:%M:%S.%f'),datetime.strftime(cend,'%Y-%m-%dT%H:%M:%S.%f'),self.sample_rate)
            #daten, dsnr, dloc = self._compute_s1(0.0, DATA.signal)
            daten, dsnr,dsnr_norm, dloc, _map = self._compute(cstart,cend, self.DATA.signal,self.DATA.station_avaliability)

            dcoord = self.lookup_table.xyz2coord(dloc)
            self.output.FileSampleRate = self.Output_SampleRate

            self._coalescenceMseed = self.output.write_decscan(self._coalescenceMseed,daten[:-1],dsnr[:-1],dsnr_norm[:-1],dcoord[:-1,:],self.sample_rate)
            
            del daten, dsnr, dsnr_norm, dloc, _map
            i += 1

        
    def _Trigger_scn(self,CoaVal,starttime,endtime):


        # Defining when exceeded threshold

        if self.NormaliseCoalescence == True:
            CoaVal['COA'] = CoaVal['COA_NORM']


        CoaVal = CoaVal[CoaVal['COA'] > self.DetectionThreshold] 
        CoaVal = CoaVal[(CoaVal['DT'] >= starttime) & (CoaVal['DT'] <= endtime)]
        
        CoaVal = CoaVal.reset_index(drop=True)


        if len(CoaVal) == 0:
            print('No Events Triggered at this threshold')
            self._no_events = True
            return


        # ----------- Determining the initial triggered events, not inspecting overlaps ------
        c = 0
        e = 1
        while c < len(CoaVal)-1:

            # Determining the index when above the level and maximum value
            d=c

            while CoaVal['DT'].iloc[d] + timedelta(seconds=1/50.0) == CoaVal['DT'].iloc[d+1]:
                d+=1
                if d+1 >= len(CoaVal)-2:
                    d=len(CoaVal)-2
                    break



            indmin = c
            indmax = d    
            indVal = np.argmax(CoaVal['COA'].iloc[np.arange(c,d+1)])


            # Determining the times for min,max and max coalescence value
            TimeMin = CoaVal['DT'].iloc[indmin]
            TimeMax = CoaVal['DT'].iloc[indmax]
            TimeVal = CoaVal['DT'].iloc[indVal]

            #print(TimeMin,TimeVal,TimeMax)

            COA_V = CoaVal['COA'].iloc[indVal]
            COA_X = CoaVal['X'].iloc[indVal]
            COA_Y = CoaVal['Y'].iloc[indVal]
            COA_Z = CoaVal['Z'].iloc[indVal]


            if (TimeVal-TimeMin) < timedelta(seconds=self.MinimumRepeat):
                TimeMin = CoaVal['DT'].iloc[indmin] + timedelta(seconds=-self.MinimumRepeat)
            if (TimeMax - TimeVal) < timedelta(seconds=self.MinimumRepeat):
                TimeMax = CoaVal['DT'].iloc[indmax] + timedelta(seconds=self.MinimumRepeat)

            
            # Appending these triggers to array
            if 'IntEvents' not in vars():
                IntEvents = pd.DataFrame([[e,TimeVal,COA_V,COA_X,COA_Y,COA_Z,TimeMin,TimeMax]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime'])
            else:
                dat       = pd.DataFrame([[e,TimeVal,COA_V,COA_X,COA_Y,COA_Z,TimeMin,TimeMax]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime'])
                IntEvents = IntEvents.append(dat,ignore_index=True)
                


            c=d+1
            e+=1



        


        # ----------- Determining the initial triggered events, not inspecting overlaps ------
        # EventNum = np.ones((len(IntEvents)),dtype=int)
        # d=1
        # for ee in range(len(IntEvents)):
        #     #if (ee+1 < len(IntEvents)) and ((IntEvents['MaxTime'].iloc[ee] - IntEvents['MinTime'].iloc[ee+1]).total_seconds() < 0):
        #     if (ee+1 < len(IntEvents)) and ((IntEvents['MinTime'].iloc[ee+1] - IntEvents['MaxTime'].iloc[ee]).total_seconds() >= self.MinimumRepeat):
        #         EventNum[ee] = d
        #         if 

        #         d+=1
        #     else:
        #         EventNum[ee] = d
        # IntEvents['EventNum'] = EventNum



        IntEvents['EventNum'] = -1
        ee = 0
        dd = 1

        if len(IntEvents['EventNum']) == 1:
            IntEvents['EventNum'].iloc[ee] = dd

        else:
            while ee+1 < len(IntEvents['EventNum']):
                if ((IntEvents['MinTime'].iloc[ee+1] - IntEvents['MaxTime'].iloc[ee]).total_seconds() >= self.MinimumRepeat):
                    IntEvents['EventNum'].iloc[ee] = dd
                    dd += 1
                    ee += 1
                    continue
                else:
                    if IntEvents['COA_V'].iloc[ee+1] > IntEvents['COA_V'].iloc[ee]:
                       IntEvents['EventNum'].iloc[ee+1] = dd 
                       IntEvents['EventNum'].iloc[ee]   = 0
                    else:
                       IntEvents['EventNum'].iloc[ee+1] = 0 
                       IntEvents['EventNum'].iloc[ee]   = dd
                    IntEvents = IntEvents[IntEvents['EventNum'] != 0]
                    IntEvents.reset_index(drop=True)


        # ----------- Combining into a single dataframe ------
        d=0
        for ee in range(1,np.max(IntEvents['EventNum'])+1):
            tmp = IntEvents[IntEvents['EventNum'] == ee].reset_index(drop=True)
            if d==0:
                EVENTS = pd.DataFrame([[ee, tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])], np.max(tmp['COA_V']),tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_Y'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_Z'].iloc[np.argmax(tmp['COA_V'])], tmp['MinTime'].iloc[np.argmax(tmp['COA_V'])], tmp['MaxTime'].iloc[np.argmax(tmp['COA_V'])]]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime'])
                d+=1
            else:
                EVENTS = EVENTS.append(pd.DataFrame([[ee, tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])], np.max(tmp['COA_V']),tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_Y'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_Z'].iloc[np.argmax(tmp['COA_V'])], tmp['MinTime'].iloc[np.argmax(tmp['COA_V'])], tmp['MaxTime'].iloc[np.argmax(tmp['COA_V'])]]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime']))






        # Defining an event id based on maximum coalescence
        EVENTS['EventID'] = EVENTS['CoaTime'].astype(str).str.replace('-','').str.replace(':','').str.replace('.','').str.replace(' ','')




        return EVENTS




    def Detect(self,starttime,endtime):
        ''' 
           Function 

           Detection of the  

        '''
        # Conduct the continious compute on the decimated grid
        self.lookup_table = self.lookup_table.decimate(self.Decimate)
        
        # Define pre-pad as a function of the onset windows
        if self.pre_pad is None:
            self.pre_pad = max(self.onset_win_p1[1],self.onset_win_s1[1]) + 3*max(self.onset_win_p1[0],self.onset_win_s1[0])
        





        # Dectect the possible events from the decimated grid
        self._continious_compute(starttime,endtime)


    def _Gaussian_Coalescence(self):
        '''
            Function to fit a gaussian for the coalescence function.
        '''


        SNR_P = self.DATA.SNR_P
        SNR_S = self.DATA.SNR_S
        X     = np.arange(SNR_P.shape[1])

        GAU_THRESHOLD = 1.4

        #---- Selecting only the data above a predefined threshold ----
        # Setting values below threshold to nan
        SNR_P[np.where(SNR_P < GAU_THRESHOLD)] = np.nan
        SNR_S[np.where(SNR_S < GAU_THRESHOLD)] = np.nan




        # Defining two blank arrays that gaussian periods should be defined for
        SNR_P_GauNum = np.zeros(SNR_P.shape)*np.nan
        SNR_S_GauNum = np.zeros(SNR_S.shape)*np.nan


        # --- Determing the indexs to fit gaussians about ---

        for s in range(len(SNR_P)):
            c=0
            e=1

            ValInd = np.where(~np.isnan(SNR_P[s,:]))[0]
            while c < len(ValInd):

                # Determining the index when above the level and maximum value
                d=c
                while ValInd[d]+1 == ValInd[d+1]:
                    d+=1
                    if d+1 >= len(ValInd)-1:
                        d=len(ValInd)-1
                        break


                indmin = c
                indmax = d  

                SNR_P_GauNum[s,ValInd[c]:ValInd[d]] = e  


                c=d+1
                e+=1

        self.DATA.SNR_P_GauNum = SNR_P_GauNum

        for s in range(len(SNR_S)):
            c=0
            e=1
            ValInd = np.where(~np.isnan(SNR_S[s,:]))[0]
            while c < len(ValInd):

                # Determining the index when above the level and maximum value
                d=c
                while ValInd[d]+1 == ValInd[d+1]:
                    d+=1
                    if d+1 >= len(ValInd)-1:
                        d=len(ValInd)-1
                        break


                indmin = c
                indmax = d  

                SNR_S_GauNum[s,ValInd[c]:ValInd[d]] = e  


                c=d+1
                e+=1



        self.DATA.SNR_S_GauNum = SNR_S_GauNum

        # --- Determing the indexs to fit gaussians about ---
        
        SNR_PGAU = np.zeros(SNR_P.shape)
        for s in range(SNR_P.shape[0]): 
            if ~np.isnan(np.nanmax(SNR_P_GauNum[s,:])):
                c=0
                for ee in range(1,int(round(np.nanmax(SNR_P_GauNum[s,:])))):


                    XSig = X[np.where((SNR_P_GauNum[s,:] == ee))[0]] 
                    YSig = SNR_P[s,np.where((SNR_P_GauNum[s,:] == ee))[0]]

                    #print(' LEN = {} and CUT_LEN = {}'.format(len(YSig),round(float(self.bp_filter_p1[0])*self.sample_rate)/10))

                    if len(YSig) > 8:

                        #self.DATA.SNR_P =  YSig

                        try:
                            lowfreq=float(self.bp_filter_p1[0])
                            p0 = [np.max(YSig), np.argmax(YSig) + np.min(XSig), 1./(lowfreq/4.)]

                            # Fitting the gaussian to the function

                            #print(XSig)
                            #print(YSig)
                            #print(p0)
                            popt, pcov = curve_fit(gaussian_func, XSig, YSig, p0) # Fit gaussian to data
                            tmp_PGau = gaussian_func(XSig.astype(float),float(popt[0]),float(popt[1]),float(popt[2]))
                            #print(tmp_PGau)

                            if c == 0:
                                SNR_P_GAU = np.zeros(X.shape)
                                SNR_P_GAU[np.where((SNR_P_GauNum[s,:] == ee))[0]] = tmp_PGau
                                c+=1
                            else:
                                SNR_P_GAU[np.where((SNR_P_GauNum[s,:] == ee))[0]] = tmp_PGau
                        except:
                            print('Error with {}'.format(ee))

                    else:
                        continue

                SNR_PGAU[s,:] =  SNR_P_GAU

        self.DATA.SNR_P = SNR_PGAU



        # --- Determing the indexs to fit gaussians about ---
        
        SNR_SGAU = np.zeros(SNR_S.shape)
        for s in range(SNR_S.shape[0]): 
            if ~np.isnan(np.nanmax(SNR_S_GauNum[s,:])):
                c=0
                for ee in range(1,int(round(np.nanmax(SNR_S_GauNum[s,:])))):


                    XSig = X[np.where((SNR_S_GauNum[s,:] == ee))[0]] 
                    YSig = SNR_S[s,np.where((SNR_S_GauNum[s,:] == ee))[0]]

                    print(' LEN = {} and CUT_LEN = {}'.format(len(YSig),round(float(self.bp_filter_p1[0])*self.sample_rate)/10))

                    if len(YSig) > 8:

                        #self.DATA.SNR_P =  YSig

                        try:
                            lowfreq=float(self.bp_filter_p1[0])
                            p0 = [np.max(YSig), np.argmax(YSig) + np.min(XSig), 1./(lowfreq/4.)]

                            # Fitting the gaussian to the function

                            print(XSig)
                            print(YSig)
                            print(p0)
                            popt, pcov = curve_fit(gaussian_func, XSig, YSig, p0) # Fit gaussian to data
                            tmp_SGau = gaussian_func(XSig.astype(float),float(popt[0]),float(popt[1]),float(popt[2]))
                            print(tmp_SGau)

                            if c == 0:
                                SNR_S_GAU = np.zeros(X.shape)
                                SNR_S_GAU[np.where((SNR_S_GauNum[s,:] == ee))[0]] = tmp_SGau
                                c+=1
                            else:
                                SNR_S_GAU[np.where((SNR_S_GauNum[s,:] == ee))[0]] = tmp_SGau
                        except:
                            print('Error with {}'.format(ee))

                    else:
                        continue

                SNR_SGAU[s,:] =  SNR_S_GAU

        self.DATA.SNR_S = SNR_PGAU


    def _GaussianTrigger(self,SNR,PHASE,cstart,eventTP,eventTS,Name,ttp,tts):
        '''
            Function to fit gaussian to onset function, based on knowledge of approximate trigger index, 
            lowest freq within signal and signal sampling rate. Will fit gaussian and return standard 
            deviation of gaussian, representative of timing error.
    
        '''

        #print('Fitting Gaussian for {} -  {} -  {}'.format(PHASE,cstart,eventT))

        sampling_rate = self.sample_rate
        trig_idx_P = int(((eventTP-cstart).seconds + (eventTP-cstart).microseconds/10.**6) *sampling_rate)
        trig_idx_S = int(((eventTS-cstart).seconds + (eventTS-cstart).microseconds/10.**6) *sampling_rate)

        # Determining the bound to search for P-wave over
        P_idxmin = int(trig_idx_P - (trig_idx_S-trig_idx_P)/(2.))
        P_idxmax = int(trig_idx_P + (trig_idx_S-trig_idx_P)/(2.))
        for ii in [P_idxmin,P_idxmax]:
            if ii < 0:
                ii = 0
            if ii > len(SNR):
                ii = len(SNR)


        lowfreq = self.bp_filter_p1[0]
        win_min = P_idxmin
        win_max = P_idxmax
        P_idxmin_new = int(trig_idx_P - int((self.MarginalWindow + ttp*self.PercentageTT)*sampling_rate))
        P_idxmax_new = int(trig_idx_P + int((self.MarginalWindow + ttp*self.PercentageTT)*sampling_rate))
        Pidxmin = np.max([P_idxmin,P_idxmin_new])
        Pidxmax = np.min([P_idxmax,P_idxmax_new])


        # Determining the bound to search for S-wave over
        S_idxmin = int(trig_idx_S - (trig_idx_S-trig_idx_P)/2.)
        S_idxmax = int(trig_idx_S + (trig_idx_S-trig_idx_P)/2.) 
        for ii in [S_idxmin,S_idxmax]:
            if ii < 0:
                ii = 0
            if ii > len(SNR):
                ii = len(SNR)

        lowfreq = self.bp_filter_s1[0]
        win_min = S_idxmin
        win_max = S_idxmax

        S_idxmin_new = int(trig_idx_S - int((self.MarginalWindow + tts*self.PercentageTT)*sampling_rate))
        S_idxmax_new = int(trig_idx_S + int((self.MarginalWindow + tts*self.PercentageTT)*sampling_rate))

        Sidxmin = np.max([S_idxmin,S_idxmin_new])
        Sidxmax = np.min([S_idxmax,S_idxmax_new])

        if PHASE == 'P':
            idxmin = Pidxmin
            idxmax = Pidxmax 
            # Determining the pick as the maximum coalescence value
            maxSNR = np.argmax(SNR[idxmin:idxmax]) + idxmin
            # Determining the trimmed SNR data to view around 
            SNR_trim = SNR[idxmin:idxmax] # or SNR
        if PHASE == 'S':
            idxmin = Sidxmin
            idxmax = Sidxmax 
            # Determining the pick as the maximum coalescence value
            maxSNR = np.argmax(SNR[idxmin:idxmax]) + idxmin
            # Determining the trimmed SNR data to view around 
            SNR_trim = SNR[idxmin:idxmax] # or SNR

        # Determining the exceedence value of the trace
        SNR_thr = SNR.copy()
        SNR_thr[Pidxmin:Pidxmax] = -1
        SNR_thr[Sidxmin:Sidxmax] = -1
        SNR_thr = SNR_thr[SNR_thr > -1]

        exceedence_value_external = np.percentile(SNR_thr,self.PickThreshold*100)
        exceedence_value_window = np.percentile(SNR_trim,88)
        exceedence_value = np.max([exceedence_value_window,exceedence_value_external]) 
        

        if SNR[maxSNR] >= exceedence_value and (SNR_trim - exceedence_value).any() > 0:
            # Using only the data around the maximum point
            exceedence = np.where((SNR_trim - exceedence_value) > 0)[0]
            exceedence_dist = np.zeros(len(exceedence))
            dd = 1
            ee = 0
            while ee < len(exceedence_dist)-1:
                if ee == len(exceedence_dist):
                    exceedence_dist[ee] = dd
                else:
                    if exceedence[ee+1] == exceedence[ee]+1:
                        exceedence_dist[ee] = dd

                    else:
                        exceedence_dist[ee] = dd
                        dd += 1
                ee += 1


            #print(exceedence_dist)
            gauidxmin = exceedence[np.where(exceedence_dist == exceedence_dist[np.argmax(SNR_trim[exceedence])])][0] + idxmin
            gauidxmax = exceedence[np.where(exceedence_dist == exceedence_dist[np.argmax(SNR_trim[exceedence])])][-1] + idxmin


            data_half_range = int(2*sampling_rate/(lowfreq)) # half range number of indices to fit guassian over (for 1 wavelengths of lowest frequency component)
            x_data = np.arange(gauidxmin, gauidxmax,dtype=float)/sampling_rate # x data, in seconds
            y_data = SNR[gauidxmin:gauidxmax] # +/- one wavelength of lowest frequency around trigger


             # Initial guess (should work for any sampling rate and frequency)
           #print(x_data, y_data, p0)



            d = 0
            for jj in range(len(x_data)):
                if d == 0:
                    XDATA = cstart + timedelta(seconds=x_data[jj])
                    d+=1
                else:
                    XDATA = np.hstack((XDATA,(cstart + timedelta(seconds=x_data[jj]))))

            try:
                p0 = [np.max(y_data), float(gauidxmin+np.argmax(y_data))/sampling_rate, data_half_range/sampling_rate]
                popt, pcov = curve_fit(gaussian_func, x_data, y_data, p0) # Fit gaussian to data
                sigma = np.absolute(popt[2]) # Get standard deviation from gaussian fit

                # Mean is popt[1]. x_data[0] + popt[1] (In seconds)
                mean = cstart + timedelta(seconds=float(popt[1]))

                maxSNR = popt[0]

                GAU_FITS = {}
                GAU_FITS['popt'] = popt
                GAU_FITS['xdata'] = x_data
                GAU_FITS['xdata_dt'] = XDATA
                GAU_FITS['PickValue'] = maxSNR
                GAU_FITS['PickThreshold'] = exceedence_value

            except:
                GAU_FITS = {}
                GAU_FITS['popt'] = 0
                GAU_FITS['xdata'] = 0
                GAU_FITS['xdata_dt'] = 0
                GAU_FITS['PickValue'] = -1
                GAU_FITS['PickThreshold'] = exceedence_value


                sigma = -1
                mean  = -1
                maxSNR = -1
        else:
            GAU_FITS = {}
            GAU_FITS['popt'] = 0
            GAU_FITS['xdata'] = 0
            GAU_FITS['xdata_dt'] = 0
            GAU_FITS['PickValue'] = -1
            GAU_FITS['PickThreshold'] = exceedence_value


            sigma = -1
            mean  = -1
            maxSNR = -1

        return GAU_FITS,maxSNR,sigma,mean





    def _ArrivalTrigger(self,EVENT_MaxCoa,EventName):
        '''
            FUNCTION - _ArrivalTrigger - Used to determine earthquake station arrival time

        '''

        SNR_P = self.DATA.SNR_P
        SNR_S = self.DATA.SNR_S

        #print(EVENT_MaxCoa[['X','Y','Z']].iloc[0])

        ttp = self.lookup_table.value_at('TIME_P', np.array(self.lookup_table.coord2xyz(np.array([EVENT_MaxCoa[['X','Y','Z']].values]))).astype(int))[0]
        tts = self.lookup_table.value_at('TIME_S', np.array(self.lookup_table.coord2xyz(np.array([EVENT_MaxCoa[['X','Y','Z']].values]))).astype(int))[0]
        

        # Determining the stations that can be picked on and the phasese
        STATION_pickS=pd.DataFrame(columns=['Name','Phase','ModelledTime','PickTime','PickError'])
        c=0
        d=0
        for s in range(len(SNR_P)):
            stationEventPT = EVENT_MaxCoa['DT'] + timedelta(seconds=ttp[s])
            stationEventST = EVENT_MaxCoa['DT'] + timedelta(seconds=tts[s])

            if self.PickingType == 'Gaussian':
                GauInfoP,maxSNR_P,Err,Mn = self._GaussianTrigger(SNR_P[s],'P',self.DATA.startTime,stationEventPT.to_pydatetime(),stationEventST.to_pydatetime(),self.lookup_table.station_data['Name'][s],ttp[s],tts[s])

            if c==0:
                GAUP = GauInfoP
                c+=1
            else:
                GAUP = np.hstack((GAUP,GauInfoP))
            
            tmpSTATION_pick = pd.DataFrame([[self.lookup_table.station_data['Name'][s],'P',stationEventPT,Mn,Err,maxSNR_P]],columns=['Name','Phase','ModelledTime','PickTime','PickError','PickSNR'])
            STATION_pickS = STATION_pickS.append(tmpSTATION_pick)


            if self.PickingType == 'Gaussian':
                GauInfoS,maxSNR_S,Err,Mn = self._GaussianTrigger(SNR_S[s],'S',self.DATA.startTime,stationEventPT.to_pydatetime(),stationEventST.to_pydatetime(),self.lookup_table.station_data['Name'][s],ttp[s],tts[s])


            if d==0:
                GAUS = GauInfoS
                d+=1
            else:
                GAUS = np.hstack((GAUS,GauInfoS))

            tmpSTATION_pick = pd.DataFrame([[self.lookup_table.station_data['Name'][s],'S',stationEventST,Mn,Err,maxSNR_S]],columns=['Name','Phase','ModelledTime','PickTime','PickError','PickSNR'])
            STATION_pickS = STATION_pickS.append(tmpSTATION_pick)

        #print(STATION_pickS)
        # Saving the output from the triggered events
        STATION_pickS = STATION_pickS[['Name','Phase','ModelledTime','PickTime','PickError','PickSNR']]
        self.output.write_stationsfile(STATION_pickS,EventName)

        return STATION_pickS,GAUP,GAUS




    def gau3d(self, nx,ny,nz,sgm):
        nx2 = (nx - 1) / 2
        ny2 = (ny - 1) / 2
        nz2 = (nz - 1) / 2
        x = np.linspace(-nx2, nx2, nx)
        y = np.linspace(-ny2, ny2, ny)
        z = np.linspace(-nz2, nz2, nz)
        ix, iy, iz = np.meshgrid(x, y, z, indexing='ij')
        if np.isscalar(sgm):
            sgm = np.repeat(sgm, 3)
        sx, sy, sz = sgm
        return np.exp(-(ix * ix) / (2 * sx * sx) - (iy * iy) / (2 * sy * sy) - (iz * iz) / (2 * sz * sz))


    def gaufilt3d(self, vol, sgm, shp=None):
        if shp is None:
            shp = vol.shape
        nx, ny, nz = shp
        flt = self.gau3d(nx, ny, nz, sgm)
        return fftconvolve(vol, flt, mode='same')


    def _mask3d(self, nn, ii, win):
        nn = np.array(nn)
        ii = np.array(ii)
        w2 = (win-1)//2
        x1, y1, z1 = np.clip(ii - w2, 0 * nn, nn)
        x2, y2, z2 = np.clip(ii + w2 + 1, 0 * nn, nn)
        mask = np.zeros(nn, dtype=np.bool)
        mask[x1:x2, y1:y2, z1:z2] = True
        return mask


    def gaufit3d(self, pdf, lx=None, ly=None, lz=None, smooth=None, thresh=0.0, win=3, mask=7):
        nx, ny, nz = pdf.shape
        if smooth is not None:
            pdf = self.gaufilt3d(pdf, smooth, [11, 11, 11])
        mx, my, mz = np.unravel_index(pdf.argmax(), pdf.shape)
        mval = pdf[mx, my, mz]
        flg = np.logical_or(
            np.logical_and(pdf > mval*np.exp(-(thresh*thresh)/2),
                           self._mask3d([nx, ny, nz], [mx, my, mz], mask)),
            self._mask3d([nx, ny, nz], [mx, my, mz], win))
        ix, iy, iz = np.where(flg)
        # print("X = {}-{}, Y = {}-{}, Z= {}-{}".format(min(ix), max(ix), min(iy), max(iy), min(iz), max(iz)))
        ncell = len(ix)

        if lx is None:
            lx = np.arange(nx)
            ly = np.arange(ny)
            lz = np.arange(nz)

        if lx.ndim == 3:
            iloc = [lx[mx, my, mz], ly[mx, my, mz], lz[mx, my, mz]]
            x = lx[ix, iy, iz] - iloc[0]
            y = ly[ix, iy, iz] - iloc[1]
            z = lz[ix, iy, iz] - iloc[2]
        else:
            iloc = [lx[mx], ly[my], lz[mz]]
            x = lx[ix] - iloc[0]
            y = ly[iy] - iloc[1]
            z = lz[iz] - iloc[2]

        X = np.c_[x * x, y * y, z * z, x * y, x * z, y * z, x, y, z, np.ones(ncell)].T
        Y = -np.log(np.clip(pdf.astype(np.float64)[ix, iy, iz], 1e-300, np.inf))

        I = np.linalg.pinv(X)
        P = np.matmul(Y, I)
        G = -np.array([2 * P[0], P[3], P[4], P[3], 2 * P[1], P[5], P[4], P[5], 2 * P[2]]).reshape((3, 3))
        H = np.array([P[6], P[7], P[8]])
        loc = np.matmul(np.linalg.inv(G), H)
        cx, cy, cz = loc
        K = P[9] - P[0] * cx * cx - P[1] * cy * cy - P[2] * cz * cz - P[3] * cx * cy - P[4] * cx * cz - P[5] * cy * cz
        M = np.array([P[0], P[3] / 2, P[4] / 2, P[3] / 2, P[1], P[5] / 2, P[4] / 2, P[5] / 2, P[2]]).reshape(3, 3)
        egv, vec = np.linalg.eig(M)
        sgm = np.sqrt(0.5 / np.clip(np.abs(egv), 1e-10, np.inf))
        val = np.exp(-K)
        csgm = np.sqrt(0.5 / np.clip(np.abs(M.diagonal()), 1e-10, np.inf))
        return loc + iloc, vec, sgm, csgm, val

    def _ErrorEllipse(self,COA3D):
        """
        Function to calculate covariance matrix and expectation hypocentre from coalescence array.
        Inputs: 
            coal_array - 3D array of coalescence values for a particular time (in x,y,z dimensions); 
            x,y,z_indices_m - 1D arrays containing labels of the indices of the coalescence grid in metres.
        Outputs are: expect_vector - x,y,z coordinates of expectation hypocentre in m; cov_matrix - Covariance matrix (of format: xx,xy,xz;yx,yy,yz;zx,zy,zz).


        """
        # samples_vectors = np.zeros((np.product(np.shape(coal_array)), 3), dtype=float)
        #samples_weights = np.zeros(np.product(np.shape(COA3D)), dtype=float)

        # Get point sample coords and weights:
        samples_weights = COA3D.flatten()

        lc = self.lookup_table.cell_count
        ly, lx, lz = np.meshgrid(np.arange(lc[1]), np.arange(lc[0]), np.arange(lc[2]))
        x_samples      = lx.flatten()*self.lookup_table.cell_size[0]
        y_samples      = ly.flatten()*self.lookup_table.cell_size[1]
        z_samples      = lz.flatten()*self.lookup_table.cell_size[2]

        SumSW = np.sum(samples_weights)

        # Calculate expectation values:
        x_expect = np.sum(samples_weights*x_samples)/SumSW
        y_expect = np.sum(samples_weights*y_samples)/SumSW
        z_expect = np.sum(samples_weights*z_samples)/SumSW
        #print('Covariance GridXYZ - X={},Y={},Z={}'.format(x_expect,y_expect,z_expect))


        # And calculate covariance matrix:
        cov_matrix = np.zeros((3,3))
        cov_matrix[0,0] = np.sum(samples_weights*(x_samples-x_expect)*(x_samples-x_expect))/SumSW
        cov_matrix[1,1] = np.sum(samples_weights*(y_samples-y_expect)*(y_samples-y_expect))/SumSW
        cov_matrix[2,2] = np.sum(samples_weights*(z_samples-z_expect)*(z_samples-z_expect))/SumSW
        cov_matrix[0,1] = np.sum(samples_weights*(x_samples-x_expect)*(y_samples-y_expect))/SumSW
        cov_matrix[1,0] = cov_matrix[0,1]
        cov_matrix[0,2] = np.sum(samples_weights*(x_samples-x_expect)*(z_samples-z_expect))/SumSW
        cov_matrix[2,0] = cov_matrix[0,2]
        cov_matrix[1,2] = np.sum(samples_weights*(y_samples-y_expect)*(z_samples-z_expect))/SumSW
        cov_matrix[2,1] = cov_matrix[1,2]



        # Determining the maximum location, and taking 2xgrid cells possitive and negative for location in each dimension
        GAU3D = self.gaufit3d(COA3D)

        # Converting the grid location to X,Y,Z
        expect_vector = self.lookup_table.xyz2coord(self.lookup_table.loc2xyz(np.array([[GAU3D[0][0],GAU3D[0][1],GAU3D[0][2]]])))[0]

        expect_vector_cov = np.array([x_expect, y_expect, z_expect], dtype=float)
        Loc_cov = self.lookup_table.xyz2coord(self.lookup_table.loc2xyz(np.array([[expect_vector_cov[0]/self.lookup_table.cell_size[0],expect_vector_cov[1]/self.lookup_table.cell_size[1],expect_vector_cov[2]/self.lookup_table.cell_size[2]]])))[0]

        ErrorXYZ = np.array([GAU3D[2][0]*self.lookup_table.cell_size[0], GAU3D[2][1]*self.lookup_table.cell_size[1], GAU3D[2][2]*self.lookup_table.cell_size[2]])


        return expect_vector, ErrorXYZ, Loc_cov, cov_matrix




    def _LocationError(self,Map4D):

        '''
            Function

        '''

        # Determining the coalescence 3D map
        CoaMap = np.log(np.sum(np.exp(Map4D),axis=-1))

        CoaMap = CoaMap/np.max(CoaMap)
        CoaMap_Cutoff = 0.88
        CoaMap[CoaMap < CoaMap_Cutoff] = CoaMap_Cutoff 
        CoaMap = CoaMap - CoaMap_Cutoff 
        CoaMap = CoaMap/np.max(CoaMap)

        # CoaMap_Cutoff = np.mean(CoaMap)
        # CoaMap[CoaMap < CoaMap_Cutoff] = CoaMap_Cutoff 
        # CoaMap = CoaMap - CoaMap_Cutoff 
        # CoaMap = CoaMap/np.max(CoaMap)

        self.CoaMAP = CoaMap


        # Determining the location error as a error-ellipse
        LOC,LOC_ERR,LOC_Cov,ErrCOV = self._ErrorEllipse(CoaMap)


        #LOC,ErrCOV = self._ErrorEllipse(CoaMap)
        LOC_ERR_Cov =  np.array([np.sqrt(ErrCOV[0,0]), np.sqrt(ErrCOV[1,1]), np.sqrt(ErrCOV[2,2])])



        # Determining maximum location and error about this point
        # ErrorVolume = np.zeros((CoaMap.shape))
        # ErrorVolume[np.where(CoaMap > self.LocationError)] = 1
        # MaxX = np.sum(np.max(np.sum(ErrorVolume,axis=0),axis=1),axis=0)*self.lookup_table.cell_size[0]
        # MaxY = np.sum(np.max(np.sum(ErrorVolume,axis=1),axis=1),axis=0)*self.lookup_table.cell_size[1]
        # MaxZ = np.sum(np.max(np.sum(ErrorVolume,axis=2),axis=1),axis=0)*self.lookup_table.cell_size[2]

        return LOC,LOC_ERR,LOC_Cov,LOC_ERR_Cov


    def TriggerSCN(self,starttime,endtime,save=False):
        CoaVal = self.output.read_scan()
        EVENTS = self._Trigger_scn(CoaVal,starttime,endtime)
        EVENTS = EVENTS[['CoaTime','COA_V']]
        EVENTS.columns = ['DateTime','CoalescenceValue']

        # saving the EVENTS file
        self.output.write_TriggerEvents(EVENTS)

        # plotting the Coalescence 
        CoaVal['DT'] = pd.to_datetime(CoaVal['DT'])
        plt.plot(CoaVal['DT'],CoaVal['COA'],'b',label='Maximum Coalescence',linewidth=1.0)

        EVENTS['DateTime'] = pd.to_datetime(EVENTS['DateTime'])
        c=0
        for i in EVENTS['DateTime']:
            if c == 0:
                plt.axvline(i.to_pydatetime() + timedelta(seconds=-(self.MinimumRepeat + self.MarginalWindow/2)),c='r', linestyle='--',label='Minimum Repeat Window',linewidth=0.5) 
                plt.axvline(i.to_pydatetime() + timedelta(seconds=(self.MinimumRepeat + self.MarginalWindow/2)),c='r', linestyle='--',linewidth=0.5)
                plt.axvline(i.to_pydatetime(),c='r',label='Detected Events',linewidth=0.5)
            else:
                plt.axvline(i.to_pydatetime() + timedelta(seconds=-(self.MinimumRepeat + self.MarginalWindow/2)),c='r', linestyle='--',linewidth=0.5)
                plt.axvline(i.to_pydatetime() + timedelta(seconds=(self.MinimumRepeat + self.MarginalWindow/2)),c='r', linestyle='--',linewidth=0.5)
                plt.axvline(i.to_pydatetime(),c='r',linewidth=0.5)
            c+=1

        plt.axhline(self.DetectionThreshold,c='g',label='Detection Threshold')
        plt.xlim([datetime.strptime(starttime,'%Y-%m-%dT%H:%M:%S.%f'),datetime.strptime(endtime,'%Y-%m-%dT%H:%M:%S.%f')])
        plt.legend()

        plt.title('Maximum Coalescence Value - {}{}'.format(self.output.path,self.output.name))
        plt.ylabel('Maximum Coalescence Value')
        plt.xlabel('DateTime')

        if save == True:
            fname = path.join(self.output.path,self.output.name + '_TriggeredEvents.pdf')
            plt.savefig(fname)
        else:
            plt.show()



    def Trigger(self,starttime,endtime,figsave=True):
        '''


        '''
        print('==============================================================================================================================')
        print('   Trigger - Triggering events from coalescence between {} and {}'.format(starttime,endtime))
        print('==============================================================================================================================')
        print('')
        print('   Parameters Specfied:')
        print('         Start Time              = {}'.format(starttime))
        print('         End   Time              = {}'.format(endtime))
        print('         Number of CPUs          = {}'.format(self.NumberOfCores))
        print('')
        print('         MarginalWindow          = {}s'.format(self.MarginalWindow))
        print('         MinimumRepeat           = {}s'.format(self.MinimumRepeat))
        print('')
        print('==============================================================================================================================')


        # Triggering Events from the 
        CoaVal = self.output.read_decscan() #self.output.read_scan()\

        #print(CoaVal)
        EVENTS = self._Trigger_scn(CoaVal,starttime,endtime)
        self.output.write_events(EVENTS)

        # Plotting the triggering
        plot_SeisScan(path.join(self.output.path,self.output.name + '.scnmseed'),self.DetectionThreshold,self.MarginalWindow,self.MinimumRepeat,starttime=starttime,endtime=endtime,savefig=figsave,stations=self.lookup_table.station_data,NormalisedPicking=self.NormaliseCoalescence)



    def Locate(self,starttime,endtime):
        '''
        

        '''

        # Intial Detection of the events from .scn file
        # CoaVal = self.output.read_decscan() #self.output.read_scan()\

        # EVENTS = self._Trigger_scn(CoaVal,starttime,endtime)

        # Changing the onset function to centred

        print('==============================================================================================================================')
        print('   Locate - Determining earthquake Location and error on less decimated grid {} and {}'.format(starttime,endtime))
        print('==============================================================================================================================')
        print('')
        print('   Parameters Specfied:')
        print('         Start Time              = {}'.format(starttime))
        print('         End   Time              = {}'.format(endtime))
        print('         Number of CPUs          = {}'.format(self.NumberOfCores))
        print('')
        print('')
        print('==============================================================================================================================')






        EVENTS = self.output.read_events(starttime,endtime)


        self._onset_centred = True

        
        # Conduct the continious compute on the decimated grid
        self.lookup_table =  self.lookup_table.decimate(self.Decimate)

        # Define pre-pad as a function of the onset windows
        if self.pre_pad is None:
            self.pre_pad = max(self.onset_win_p1[1],self.onset_win_s1[1]) + 3*max(self.onset_win_p1[0],self.onset_win_s1[0])

        #
        Triggered = pd.DataFrame(columns=['DT','COA','X','Y','Z','ErrX','ErrY','ErrZ'])
        for e in range(len(EVENTS)):

            print('--Processing for Event {} of {} - {}'.format(e+1,len(EVENTS),(EVENTS['EventID'].iloc[e])))
            #tic()

            # WTF WTF WTF
            # Adding the window for the tapering 
            # timeLen = self.pre_pad + self.post_pad + (EVENTS['MaxTime'].iloc[e] - EVENTS['MinTime'].iloc[e]).total_seconds()
            # self.pre_pad = self.pre_pad + round(timeLen*0.05)
            # self.post_pad = self.post_pad + round(timeLen*0.05)
            # print(self.pre_pad,self.post_pad)

            # Determining the Seismic event location
            cstart = EVENTS['CoaTime'].iloc[e] + 2*timedelta(seconds=-self.MarginalWindow)  + timedelta(seconds=-self.pre_pad) 
            cend   = EVENTS['CoaTime'].iloc[e] + 2*timedelta(seconds=self.MarginalWindow)   + timedelta(seconds=self.post_pad)

            self.DATA.read_mseed(cstart.strftime('%Y-%m-%dT%H:%M:%S.%f'),cend.strftime('%Y-%m-%dT%H:%M:%S.%f'),self.sample_rate)

            daten, dsnr, dsnr_norm, dloc, self.MAP = self._compute(cstart,cend,self.DATA.signal,self.DATA.station_avaliability)
            dcoord = self.lookup_table.xyz2coord(np.array(dloc).astype(int))
            EventCoaVal = pd.DataFrame(np.array((daten,dsnr,dcoord[:,0],dcoord[:,1],dcoord[:,2])).transpose(),columns=['DT','COA','X','Y','Z'])
            EventCoaVal['DT'] = pd.to_datetime(EventCoaVal['DT'])
            self.EVENT = EventCoaVal
            self.EVENT_max = self.EVENT.iloc[self.EVENT['COA'].astype('float').idxmax()]

            cstart_mw = self.EVENT_max['DT'] + timedelta(seconds=-self.MarginalWindow) 
            cend_mw   = self.EVENT_max['DT'] + timedelta(seconds=self.MarginalWindow)

            # Marginal window not within windo in question - Recompute until the time window is correct. 
            while (np.min(self.EVENT['DT'])  > cstart_mw) or (np.max(self.EVENT['DT']) < cend_mw):
                cstart = self.EVENT_max['DT'] + 2*timedelta(seconds=-self.MarginalWindow)  + timedelta(seconds=-self.pre_pad) 
                cend   = self.EVENT_max['DT'] + 2*timedelta(seconds=self.MarginalWindow)   + timedelta(seconds=self.post_pad)

                self.DATA.read_mseed(cstart.strftime('%Y-%m-%dT%H:%M:%S.%f'),cend.strftime('%Y-%m-%dT%H:%M:%S.%f'),self.sample_rate)

                daten, dsnr, dsnr_norm, dloc, self.MAP = self._compute(cstart,cend,self.DATA.signal,self.DATA.station_avaliability)
                dcoord = self.lookup_table.xyz2coord(np.array(dloc).astype(int))
                EventCoaVal = pd.DataFrame(np.array((daten,dsnr,dcoord[:,0],dcoord[:,1],dcoord[:,2])).transpose(),columns=['DT','COA','X','Y','Z'])
                EventCoaVal['DT'] = pd.to_datetime(EventCoaVal['DT'])
                self.EVENT = EventCoaVal
                self.EVENT_max = self.EVENT.iloc[self.EVENT['COA'].astype('float').idxmax()]
                cstart_mw = self.EVENT_max['DT'] + timedelta(seconds=-self.MarginalWindow) 
                cend_mw   = self.EVENT_max['DT'] + timedelta(seconds=self.MarginalWindow)

            # Specifying the 
            self.EVENT = self.EVENT[(self.EVENT['DT'] >= cstart_mw) & (self.EVENT['DT'] <= cend_mw) ]
            self.MAP = self.MAP[:,:,:,self.EVENT.index[0]:self.EVENT.index[-1]]
            self.EVENT = self.EVENT.reset_index(drop=True)
            self.EVENT_max = self.EVENT.iloc[self.EVENT['COA'].astype('float').idxmax()]


            # Determining the hypocentral location from the maximum over the marginal window.
            Picks,GAUP,GAUS = self._ArrivalTrigger(self.EVENT_max,(EVENTS['EventID'].iloc[e]))

            StationPick = {}
            StationPick['Pick'] = Picks
            StationPick['GAU_P'] = GAUP
            StationPick['GAU_S'] = GAUS
            #toc()

            # Determining earthquake location error
            #tic()
            LOC,LOC_ERR,LOC_Cov,LOC_ERR_Cov = self._LocationError(self.MAP)
            #toc()


            EV = pd.DataFrame([[self.EVENT_max['DT'],self.EVENT_max['COA'],EVENTS['COA_V'].iloc[e],self.EVENT_max['X'],self.EVENT_max['Y'],self.EVENT_max['Z'],LOC[0],LOC[1],LOC[2],LOC_ERR[0],LOC_ERR[1],LOC_ERR[2],LOC_Cov[0],LOC_Cov[1],LOC_Cov[2],LOC_ERR_Cov[0],LOC_ERR_Cov[1],LOC_ERR_Cov[2]]],columns=['DT','COA','DecCOA','X','Y','Z','Gaussian_X','Gaussian_Y','Gaussian_Z','Gaussian_ErrX','Gaussian_ErrY','Gaussian_ErrZ','Covariance_X','Covariance_Y','Covariance_Z','Covariance_ErrX','Covariance_ErrY','Covariance_ErrZ'])
            self.output.write_event(EV,str(EVENTS['EventID'].iloc[e]))
            if self.CutMSEED == True:
                print('Creating cut Mini-SEED')
                #tic()
                self.output.cut_mseed(self.DATA,str(EVENTS['EventID'].iloc[e]))
                #toc()

            # Outputting coalescence grids and triggered events
            if self.CoalescenceTrace == True:
                #tic()
                print('Creating Station Traces')
                SeisPLT = SeisPlot(self.lookup_table,self.MAP,self.CoaMAP,self.DATA,self.EVENT,StationPick,self.MarginalWindow)
                SeisPLT.CoalescenceTrace(SaveFilename='{}_{}'.format(path.join(self.output.path, self.output.name),EVENTS['EventID'].iloc[e]))
                #toc()

            if self.CoalescenceGrid == True:
                tic()
                print('Creating 4D Coalescence Grids')
                self.output.write_coal4D(self.MAP,EVENTS['EventID'].iloc[e],cstart,cend)
                toc()

            if self.CoalescenceVideo == True:
                #tic()
                print('Creating Seismic Videos')
                SeisPLT = SeisPlot(self.lookup_table,self.MAP,self.CoaMAP,self.DATA,self.EVENT,StationPick,self.MarginalWindow)
                SeisPLT.CoalescenceVideo(SaveFilename='{}_{}'.format(path.join(self.output.path, self.output.name),EVENTS['EventID'].iloc[e]))
                #toc()

            if self.CoalescencePicture == True:
                #tic()
                print('Creating Seismic Picture')
                SeisPLT = SeisPlot(self.lookup_table,self.MAP,self.CoaMAP,self.DATA,self.EVENT,StationPick,self.MarginalWindow)
                SeisPLT.CoalescenceMarginal(SaveFilename='{}_{}'.format(path.join(self.output.path, self.output.name),EVENTS['EventID'].iloc[e]),Earthquake=EV)
                #toc()

            self.MAP    = None
            self.CoaMAP = None
            self.EVENT  = None
            self.cstart = None
            self.cend   = None

        self._onset_centred = False



class plot_SeisScan:

    def __init__(self,file,DetectionThreshold,MarginalWindow,MinimumRepeat,starttime=None,endtime=None,savefig=False,stations=None,NormalisedPicking=False):



        # Defining the input arguments
        self.file                 = file
        self.DetectionThreshold   = DetectionThreshold
        self.MarginalWindow       = MarginalWindow
        self.MinimumRepeat        = MinimumRepeat
        self.startTime            = starttime
        self.endTime              = endtime 
        self.savefig              = savefig
        self.stations             = stations
        self._no_events           = False

        # Defining the events
        self.Events               = None


        # Determining the filename
        self.fname = ''.join(file.split('.')[:-1]) + '_Trigger.pdf'



        # Reading the Coalescence file and determining the samplerate of the data 
        COA = obspy.read(file)
        sampling_rate = COA.select(station='COA')[0].stats.sampling_rate
        DATA = pd.DataFrame()
        DATA['DT']       = np.arange(datetime.strptime(str(COA.select(station='COA')[0].stats.starttime),'%Y-%m-%dT%H:%M:%S.%fZ'),datetime.strptime(str(COA.select(station='COA')[0].stats.endtime),'%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(seconds=1/COA.select(station='COA')[0].stats.sampling_rate) ,timedelta(seconds=1/COA.select(station='COA')[0].stats.sampling_rate))
        DATA['COA']      = COA.select(station='COA')[0].data/1e8
        DATA['COA_NORM'] = COA.select(station='COA_N')[0].data/1e8
        DATA['X']        = COA.select(station='X')[0].data/1e6
        DATA['Y']        = COA.select(station='Y')[0].data/1e6
        DATA['Z']        = COA.select(station='Z')[0].data

        # If starttime and endtime are not defined then determine it from the station information
        if self.startTime == None:
           self.startTime = np.min(DATA['DT'])
        else:   
           self.startTime = datetime.strptime(self.startTime,'%Y-%m-%dT%H:%M:%S.%f')

        if self.endTime == None:
           self.endTime = np.max(DATA['DT'])
        else:   
           self.endTime = datetime.strptime(self.endTime,'%Y-%m-%dT%H:%M:%S.%f')


        self.DATA = DATA

        if NormalisedPicking == True:
            self.CoaType = 'COA_NORM'
        else:
            self.CoaType = 'COA'

        # Determining the samplerate of the data 
        self.sample_rate = int(1/(DATA['DT'].iloc[1] - DATA['DT'].iloc[0]).total_seconds())


        # Running the plotting function
        self._plot_scn()


    def _Trigger_scn(self):

        '''
            Determining the triggers from the .scn file and parameters given

        '''

        starttime = self.startTime 
        endtime   = self.endTime


        # Defining when exceeded threshold
        if self.CoaType == 'COA':
            CoaVal = self.DATA[['DT','COA','X','Y','Z']]
        else:
            CoaVal = self.DATA.copy()
            CoaVal['COA'] = CoaVal['COA_NORM']
            CoaVal = CoaVal[['DT','COA','X','Y','Z']]


        CoaVal = CoaVal[CoaVal['COA'] > self.DetectionThreshold] 
        CoaVal = CoaVal[(CoaVal['DT'] >= starttime) & (CoaVal['DT'] <= endtime)]
        
        CoaVal = CoaVal.reset_index(drop=True)


        if len(CoaVal) == 0:
            print('No Events Triggered at this threshold')
            self._no_events = True
            return

        # ----------- Determining the initial triggered events, not inspecting overlaps ------
        c = 0
        e = 1
        while c < len(CoaVal)-1:

            # Determining the index when above the level and maximum value
            d=c

            while CoaVal['DT'].iloc[d] + timedelta(seconds=1/50.0) == CoaVal['DT'].iloc[d+1]:
                d+=1
                if d+1 >= len(CoaVal)-2:
                    d=len(CoaVal)-2
                    break



            indmin = c
            indmax = d    
            indVal = np.argmax(CoaVal['COA'].iloc[np.arange(c,d+1)])


            # Determining the times for min,max and max coalescence value
            TimeMin = CoaVal['DT'].iloc[indmin]
            TimeMax = CoaVal['DT'].iloc[indmax]
            TimeVal = CoaVal['DT'].iloc[indVal]

            #print(TimeMin,TimeVal,TimeMax)

            COA_V = CoaVal['COA'].iloc[indVal]
            COA_X = CoaVal['X'].iloc[indVal]
            COA_Y = CoaVal['Y'].iloc[indVal]
            COA_Z = CoaVal['Z'].iloc[indVal]


            if (TimeVal-TimeMin) < timedelta(seconds=self.MinimumRepeat):
                TimeMin = CoaVal['DT'].iloc[indmin] + timedelta(seconds=-self.MinimumRepeat)
            if (TimeMax - TimeVal) < timedelta(seconds=self.MinimumRepeat):
                TimeMax = CoaVal['DT'].iloc[indmax] + timedelta(seconds=self.MinimumRepeat)

            
            # Appending these triggers to array
            if 'IntEvents' not in vars():
                IntEvents = pd.DataFrame([[e,TimeVal,COA_V,COA_X,COA_Y,COA_Z,TimeMin,TimeMax]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime'])
            else:
                dat       = pd.DataFrame([[e,TimeVal,COA_V,COA_X,COA_Y,COA_Z,TimeMin,TimeMax]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime'])
                IntEvents = IntEvents.append(dat,ignore_index=True)
                


            c=d+1
            e+=1



        


        # ----------- Determining the initial triggered events, not inspecting overlaps ------
        # EventNum = np.ones((len(IntEvents)),dtype=int)
        # d=1
        # for ee in range(len(IntEvents)):
        #     #if (ee+1 < len(IntEvents)) and ((IntEvents['MaxTime'].iloc[ee] - IntEvents['MinTime'].iloc[ee+1]).total_seconds() < 0):
        #     if (ee+1 < len(IntEvents)) and ((IntEvents['MinTime'].iloc[ee+1] - IntEvents['MaxTime'].iloc[ee]).total_seconds() >= self.MinimumRepeat):
        #         EventNum[ee] = d
        #         if 

        #         d+=1
        #     else:
        #         EventNum[ee] = d
        # IntEvents['EventNum'] = EventNum

        IntEvents['EventNum'] = -1
        ee = 0
        dd = 1

        if len(IntEvents['EventNum']) == 1:
            IntEvents['EventNum'].iloc[ee] = dd

        else:
            while ee+1 < len(IntEvents['EventNum']):
                if ((IntEvents['MinTime'].iloc[ee+1] - IntEvents['MaxTime'].iloc[ee]).total_seconds() >= self.MinimumRepeat):
                    IntEvents['EventNum'].iloc[ee] = dd
                    dd += 1
                    ee += 1
                    continue
                else:
                    if IntEvents['COA_V'].iloc[ee+1] > IntEvents['COA_V'].iloc[ee]:
                       IntEvents['EventNum'].iloc[ee+1] = dd 
                       IntEvents['EventNum'].iloc[ee]   = 0
                    else:
                       IntEvents['EventNum'].iloc[ee+1] = 0 
                       IntEvents['EventNum'].iloc[ee]   = dd
                    IntEvents = IntEvents[IntEvents['EventNum'] != 0]
                    IntEvents.reset_index(drop=True)

        # ----------- Combining into a single dataframe ------
        d=0
        for ee in range(1,np.max(IntEvents['EventNum'])+1):
            tmp = IntEvents[IntEvents['EventNum'] == ee].reset_index(drop=True)
            if d==0:
                EVENTS = pd.DataFrame([[ee, tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])], np.max(tmp['COA_V']),tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_Y'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_Z'].iloc[np.argmax(tmp['COA_V'])], tmp['MinTime'].iloc[np.argmax(tmp['COA_V'])], tmp['MaxTime'].iloc[np.argmax(tmp['COA_V'])]]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime'])
                d+=1
            else:
                EVENTS = EVENTS.append(pd.DataFrame([[ee, tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])], np.max(tmp['COA_V']),tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_Y'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_Z'].iloc[np.argmax(tmp['COA_V'])], tmp['MinTime'].iloc[np.argmax(tmp['COA_V'])], tmp['MaxTime'].iloc[np.argmax(tmp['COA_V'])]]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime']))






        # Defining an event id based on maximum coalescence
        EVENTS['EventID'] = EVENTS['CoaTime'].astype(str).str.replace('-','').str.replace(':','').str.replace('.','').str.replace(' ','')





        self.Events = EVENTS


    def _plot_scn(self):

        # Determining the events to plot

        self._Trigger_scn()
        if self._no_events == True:
            return

        fig = plt.figure(figsize=(30,15))
        fig.patch.set_facecolor('white')
        plt_Coa       =  plt.subplot2grid((6,16), (0, 0), colspan=9,rowspan=3)
        plt_Coa_Norm  =  plt.subplot2grid((6,16), (3, 0), colspan=9,rowspan=3,sharex=plt_Coa)
        plt_Map       =  plt.subplot2grid((6,16), (0, 10), colspan=4,rowspan=4)
        plt_XZ        =  plt.subplot2grid((6,16), (4, 10), colspan=4,rowspan=2,sharex=plt_Map)
        plt_YZ        =  plt.subplot2grid((6,16), (0, 14), colspan=2,rowspan=4,sharey=plt_Map)

        # Setting the axis names


        # Plotting the Traces
        plt_Coa.plot(self.DATA['DT'],self.DATA['COA'],'b',label='Maximum Coalescence',linewidth=1.0)
        plt_Coa.get_xaxis().set_ticks([])
        plt_Coa_Norm.plot(self.DATA['DT'],self.DATA['COA_NORM'],'b',label='Maximum Coalescence',linewidth=1.0)

        self.Events['CoaTime'] = pd.to_datetime(self.Events['CoaTime'])
        self.Events['MinTime'] = pd.to_datetime(self.Events['MinTime'])
        self.Events['MaxTime'] = pd.to_datetime(self.Events['MaxTime'])
        c=0
        for i in range(len(self.Events)):
            if c == 0:

                plt_Coa.axvspan(self.Events['MinTime'].iloc[i].to_pydatetime() + timedelta(seconds=-self.MinimumRepeat),self.Events['MaxTime'].iloc[i].to_pydatetime() + timedelta(seconds=self.MinimumRepeat),label='Minimum Repeat Window',alpha=0.5, color='red')
                plt_Coa.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime() + timedelta(seconds=-self.MarginalWindow),c='m', linestyle='--',label='Marginal Window',linewidth=1.75) 
                plt_Coa.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime() + timedelta(seconds=self.MarginalWindow),c='m', linestyle='--',linewidth=1.75)
                plt_Coa.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime(),c='m',label='Detected Events',linewidth=1.75)

                plt_Coa_Norm.axvspan(self.Events['MinTime'].iloc[i].to_pydatetime() + timedelta(seconds=-self.MinimumRepeat),self.Events['MaxTime'].iloc[i].to_pydatetime() + timedelta(seconds=self.MinimumRepeat),label='Minimum Repeat Window',alpha=0.5, color='red')
                plt_Coa_Norm.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime() + timedelta(seconds=-self.MarginalWindow),c='m', linestyle='--',label='Marginal Window',linewidth=1.75) 
                plt_Coa_Norm.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime() + timedelta(seconds=self.MarginalWindow),c='m', linestyle='--',linewidth=1.75)
                plt_Coa_Norm.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime(),c='m',label='Detected Events',linewidth=1.75)
            else:
                plt_Coa.axvspan(self.Events['MinTime'].iloc[i].to_pydatetime() + timedelta(seconds=-self.MinimumRepeat),self.Events['MaxTime'].iloc[i].to_pydatetime() + timedelta(seconds=self.MinimumRepeat),alpha=0.5, color='red')
                plt_Coa.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime() + timedelta(seconds=-self.MarginalWindow),c='m', linestyle='--',linewidth=1.75) 
                plt_Coa.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime() + timedelta(seconds=self.MarginalWindow),c='m', linestyle='--',linewidth=1.75)
                plt_Coa.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime(),c='m',linewidth=1.75)

                plt_Coa_Norm.axvspan(self.Events['MinTime'].iloc[i].to_pydatetime() + timedelta(seconds=-self.MinimumRepeat),self.Events['MaxTime'].iloc[i].to_pydatetime() + timedelta(seconds=self.MinimumRepeat),alpha=0.5, color='red')
                plt_Coa_Norm.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime() + timedelta(seconds=-self.MarginalWindow),c='m', linestyle='--',linewidth=1.75) 
                plt_Coa_Norm.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime() + timedelta(seconds=self.MarginalWindow),c='m', linestyle='--',linewidth=1.75)
                plt_Coa_Norm.axvline(self.Events['CoaTime'].iloc[i].to_pydatetime(),c='m',linewidth=1.75)
            c+=1


        # Plotting the detection threshold and labels
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt_Coa.set_xlim([self.startTime,self.endTime])
        plt_Coa.text(.5,.9,'MaximumCoalescence',horizontalalignment='center',transform=plt_Coa.transAxes,bbox=props)
        plt_Coa.legend(loc=2)
        plt_Coa.set_ylabel('Maximum Coalescence Value')
        plt_Coa_Norm.text(.5,.9,'Normalised Maximum Coalescence',horizontalalignment='center',transform=plt_Coa_Norm.transAxes,bbox=props)
        plt_Coa_Norm.set_xlim([self.startTime,self.endTime])
        plt_Coa_Norm.legend(loc=2)
        plt_Coa_Norm.set_ylabel('Normalised Maximum Coalescence Value')
        plt_Coa_Norm.set_xlabel('DateTime')

        if self.CoaType == 'COA':
            plt_Coa.axhline(self.DetectionThreshold,c='g',label='Detection Threshold')
        else:
            plt_Coa_Norm.axhline(self.DetectionThreshold,c='g',label='Detection Threshold')


        # Plotting the scatter of the earthquake locations
        plt_Map.set_title('Decimated Coalescence Earthquake Locations')
        plt_Map.scatter(self.Events['COA_X'],self.Events['COA_Y'],50,self.Events['COA_V'])
        plt_Map.get_xaxis().set_ticks([])
        plt_Map.get_yaxis().set_ticks([])


        plt_YZ.scatter(self.Events['COA_Z'],self.Events['COA_Y'],50,self.Events['COA_V'])
        plt_YZ.yaxis.tick_right()
        plt_YZ.yaxis.set_label_position("right")
        plt_YZ.invert_xaxis()
        plt_YZ.set_ylabel('Latitude (deg)')
        plt_YZ.set_xlabel('Depth (m)')



        plt_XZ.scatter(self.Events['COA_X'],self.Events['COA_Z'],50,self.Events['COA_V'])
        plt_XZ.yaxis.tick_right()
        plt_XZ.yaxis.set_label_position("right")
        plt_XZ.set_xlabel('Longitude (deg)')
        plt_XZ.set_ylabel('Depth (m)')


        # Plotting stations if file given
        if self.stations != None:
            plt_Map.scatter(self.stations['Longitude'],self.stations['Latitude'],20,)

            # Plotting the station locations
            plt_Map.scatter(self.stations['Longitude'],self.stations['Latitude'],15,marker='^',color='black')
            plt_XZ.scatter(self.stations['Longitude'],self.stations['Elevation'],15,marker='^',color='black')
            plt_YZ.scatter(self.stations['Elevation'],self.stations['Latitude'],15,marker='<',color='black')
            for i,txt in enumerate(self.stations['Name']):
                plt_Map.annotate(txt,[self.stations['Longitude'][i],self.stations['Latitude'][i]],color='black')


        if self.savefig == True:
            plt.savefig(self.fname)
        else:
            plt.show()




############################

# class DeepLearningPhaseDetection:

#     def __init__(self,sign,sige,sigz,srate):


#         #####################
#         # Hyperparameters
#         self.freq_min = 2.0
#         self.freq_max = 16.0
        
#         self.decimate_data = False # If false, assumes data is already 100 Hz samprate

#         self.n_shift = 10 # Number of samples to shift the sliding window at a time
#         self.n_gpu   = 0 # Number of GPUs to use (if any)


#         self.batch_size = 1000*3

#         self.half_dur = 2.00
#         self.only_dt  = 0.01
#         self.n_win    = int(self.half_dur/self.only_dt)
#         self.n_feat   = 2*self.n_win

#         self.sign       = sign
#         self.sige       = sige
#         self.sigz       = sigz
#         self.srate      = srate

#         self.prob_S  = None
#         self.prob_P  = None
#         self.prob_N  = None 

#         self.models_path  = '/raid1/jds70/PhaseLink/generalized-phase-detection/model_pol.json'
#         self.weights_path = '/raid1/jds70/PhaseLink/generalized-phase-detection/model_pol_best.hdf5'


#         self.PhaseProbability()


#     def sliding_window(self,data, size, stepsize=1, padded=False, axis=-1, copy=True):
#         """
#         Calculate a sliding window over a signal
#         Parameters
#         ----------
#         data : numpy array
#             The array to be slided over.
#         size : int
#             The sliding window size
#         stepsize : int
#             The sliding window stepsize. Defaults to 1.
#         axis : int
#             The axis to slide over. Defaults to the last axis.
#         copy : bool
#             Return strided array as copy to avoid sideffects when manipulating the
#             output array.
#         Returns
#         -------
#         data : numpy array
#             A matrix where row in last dimension consists of one instance
#             of the sliding window.
#         Notes
#         -----
#         - Be wary of setting `copy` to `False` as undesired sideffects with the
#           output values may occurr.
#         Examples
#         --------
#         >>> a = numpy.array([1, 2, 3, 4, 5])
#         >>> sliding_window(a, size=3)
#         array([[1, 2, 3],
#                [2, 3, 4],
#                [3, 4, 5]])
#         >>> sliding_window(a, size=3, stepsize=2)
#         array([[1, 2, 3],
#                [3, 4, 5]])
#         See Also
#         --------
#         pieces : Calculate number of pieces available by sliding
#         """
#         if axis >= data.ndim:
#             raise ValueError(
#                 "Axis value out of range"
#             )

#         if stepsize < 1:
#             raise ValueError(
#                 "Stepsize may not be zero or negative"
#             )

#         if size > data.shape[axis]:
#             raise ValueError(
#                 "Sliding window size may not exceed size of selected axis"
#             )

#         shape = list(data.shape)
#         shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
#         shape.append(size)

#         strides = list(data.strides)
#         strides[axis] *= stepsize
#         strides.append(data.strides[axis])

#         strided = np.lib.stride_tricks.as_strided(
#             data, shape=shape, strides=strides
#         )

#         if copy:
#             return strided.copy()
#         else:
#             return strided


#     def PhaseProbability(self):

#         # load json and create model
#         json_file = open(self.models_path, 'r')
#         loaded_model_json = json_file.read()
#         json_file.close()
#         model = model_from_json(loaded_model_json, custom_objects={'tf':tf})

#         # load weights into new model
#         model.load_weights(self.weights_path)
#         #print("Loaded model from disk")

#         # Parallesing for GPU usage
#         if self.n_gpu > 1:
#             from keras.utils import multi_gpu_model
#             model = multi_gpu_model(model, gpus=self.n_gpu)


#         #Manipulating the streams so samplerate and station same
#         sr  = self.srate
#         dt  = 1.0/sr

#         self.prob_S = np.zeros(self.sign.shape)
#         self.prob_P = np.zeros(self.sign.shape)
#         self.prob_N = np.zeros(self.sign.shape)

#         for ii in range(self.sign.shape[0]):
#             tt = (np.arange(0, self.sign.shape[1], self.n_shift) + self.n_win) * dt
#             tt_i = np.arange(0, self.sign.shape[1], self.n_shift) + self.n_feat

#             sliding_N = self.sliding_window(self.sign[ii,:], self.n_feat, stepsize=self.n_shift)
#             sliding_E = self.sliding_window(self.sige[ii,:], self.n_feat, stepsize=self.n_shift)
#             sliding_Z = self.sliding_window(self.sigz[ii,:], self.n_feat, stepsize=self.n_shift)

#             tr_win        = np.zeros((sliding_N.shape[0], self.n_feat, 3))
#             tr_win[:,:,0] = sliding_N
#             tr_win[:,:,1] = sliding_E
#             tr_win[:,:,2] = sliding_Z
#             #tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
#             tt = tt[:tr_win.shape[0]]
#             tt_i = tt_i[:tr_win.shape[0]]

#             ts = model.predict(tr_win, verbose=False, batch_size=tr_win.shape[0])

#             SS = np.interp(np.arange(self.sign.shape[1])*dt,tt,ts[:,1])
#             PP = np.interp(np.arange(self.sign.shape[1])*dt,tt,ts[:,0])
#             NN = np.interp(np.arange(self.sign.shape[1])*dt,tt,ts[:,2])
#             PP[np.isnan(PP)] = 0.0
#             SS[np.isnan(SS)] = 0.0
#             NN[np.isnan(NN)] = 0.0

#             #plt.plot(ts[:,0])
#             #plt.savefig('TEST.pdf')


#             self.prob_S[ii,:] = SS
#             self.prob_P[ii,:] = PP
#             self.prob_N[ii,:] = NN





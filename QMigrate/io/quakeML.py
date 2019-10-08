import warnings
import numpy as np
from obspy import UTCDateTime, Stream, Trace
from obspy.signal.invsim import cosine_taper
from obspy.signal.trigger import classic_sta_lta
import pandas as pd
from scipy.interpolate import Rbf
from scipy.optimize import curve_fit
from scipy.signal import butter, lfilter, fftconvolve
import sys
import torch
from obspy.signal.trigger import trigger_onset
from scipy.interpolate import interp1d

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class UNet(torch.nn.Module):

    def __init__(self, num_channels=3, num_classes=3):
        super(UNet, self).__init__()
        from torch.nn import MaxPool1d, Conv1d, ConvTranspose1d
        self.relu = torch.nn.ReLU()
        self.maxpool = MaxPool1d(kernel_size=2, stride=2)
        self.conv11 = Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.conv12 = Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.conv21 = Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv22 = Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.conv31 = Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv32 = Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.conv41 = Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv42 = Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm1d(512)

        self.conv51 = Conv1d(512, 1024, kernel_size=3, padding=1)
        self.conv52 = Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm1d(1024)

        self.uconv6 = ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.conv61 = Conv1d(1024, 512, kernel_size=3, padding=1)
        self.conv62 = Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn6 = torch.nn.BatchNorm1d(512)

        self.uconv7 = ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv71 = Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv72 = Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn7 = torch.nn.BatchNorm1d(256)

        self.uconv8 = ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv81 = Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv82 = Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn8 = torch.nn.BatchNorm1d(128)

        self.uconv9 = ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv91 = Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv92 = Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn9 = torch.nn.BatchNorm1d(64)

        self.conv93 = Conv1d(64, num_classes, kernel_size=1, padding=0)


    def forward(self, x):
        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x1d = self.relu(x)
        #x1d = self.bn1(x1d)
        x = self.maxpool(x1d)

        x = self.conv21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x2d = self.relu(x)
        #x2d = self.bn2(x2d)
        x = self.maxpool(x2d)

        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x3d = self.relu(x)
        #x3d = self.bn3(x3d)
        x = self.maxpool(x3d)

        x = self.conv41(x)
        x = self.relu(x)
        x = self.conv42(x)
        x4d = self.relu(x)
        #x4d = self.bn4(x4d)
        x = self.maxpool(x4d)

        x = self.conv51(x)
        x = self.relu(x)
        x = self.conv52(x)
        x5d = self.relu(x)
        #x5d = self.bn5(x5d)

        x6u = self.uconv6(x5d)
        x = torch.cat((x4d, x6u), 1)
        x = self.conv61(x)
        x = self.relu(x)
        x = self.conv62(x)
        x = self.relu(x)
        #x = self.bn6(x)

        x7u = self.uconv7(x)
        x = torch.cat((x3d, x7u), 1)
        x = self.conv71(x)
        x = self.relu(x)
        x = self.conv72(x)
        x = self.relu(x)
        #x = self.bn7(x)

        x8u = self.uconv8(x)
        x = torch.cat((x2d, x8u), 1)
        x = self.conv81(x)
        x = self.relu(x)
        x = self.conv82(x)
        x = self.relu(x)
        #x = self.bn8(x)

        x9u = self.uconv9(x)
        x = torch.cat((x1d, x9u), 1)
        x = self.conv91(x)
        x = self.relu(x)
        x = self.conv92(x)
        x = self.relu(x)
        #x = self.bn9(x)

        x = self.conv93(x)

        x = torch.sigmoid(x)
        return x


class GPD:
    def __init__(self,signal,sampling_rate,stdError,params=None):

        # Defining the input data
        self.signal        = signal
        self.sampling_rate = sampling_rate
        self.stdError      = stdError*self.sampling_rate

        if params == None:
            self.params ={"min_proba": 0.01,
                     "min_trig_dur": 0.0,
                     "model_file_S": "/atomic-data/zross/cahuilla/gpd_models/model_S_021_0.692961.pt",
                     "model_file_P": "/atomic-data/zross/cahuilla/gpd_models/model_P_015_0.691434.pt",
                     "batch_size0000000": 16384,
                     "batch_size": 4096,
                     "device": "cuda:6",
                     "half_dur": 8.0,
                     "delta": 0.01
                    }

        # Defining variables from input data
        self.onset  = np.zeros((2,self.signal.shape[1],self.signal.shape[2]))
        self.device = torch.device(self.params['device'])
        self.n_win  = int(self.params['half_dur']/self.params['delta'])
        self.n_feat = 2*self.n_win


        # If params batch size is larger than singal size then reduce parameters
        if self.params['batch_size'] > self.signal.shape[1]:
           self.params['batch_size'] = self.signal.shape[1]


        # Resampling the data to the user defined sample rate
        if self.sampling_rate != 1/self.params['delta']:
            self.signal = self.resample_data(self.signal)


        # Loading the models for sesimic detection
        if 'model_file_S' not in self.params.keys() and 'model_file_P' not in self.params.keys():
            self.model_P = self._train_dataset(self.params['model_file_P'])
            self.model_S = self._train_dataset(self.params['model_file_P'])

        else:
            self.model_P = self._load_trained_model(self.params['model_file_P']).to(self.device)
            self.model_S = self._load_trained_model(self.params['model_file_S']).to(self.device)


        # Creating a gaussian function to fit with 
        self.GauX = np.arange(-round(self.stdError)*3,round(self.stdError)*3)
        self.GauY = gaussian(np.arange(-round(self.stdError)*3,round(self.stdError)*3),0,round(self.stdError))


    def _train_dataset(self):
        '''

        '''

        print("Function not complete yet. Please come back soon ! :) Jonny")
        sys.exit()

    def _load_trained_model(self,model_file):

        model = UNet(num_channels=3, num_classes=1).to(self.device)
        #model = torch.nn.DataParallel(model)
        checkpoint = torch.load(model_file,map_location=self.device) #BQL
        #checkpoint = torch.load(model_file)
        state_dict = {}
        for key in checkpoint['model_state_dict']:
            if "tracked" in key:
                continue
            state_dict[key] = checkpoint['model_state_dict'][key]
        model.load_state_dict(state_dict)
        model.eval()
        return model





    def sliding_window(self,data, size, stepsize=None, padded=False, axis=-1, copy=True):
        """
        Calculate a sliding window over a signal
        Parameters
        ----------
        data : numpy array
            The array to be slided over.
        size : int
            The sliding window size
        stepsize : int
            The sliding window stepsize. Defaults to 1.
        axis : int
            The axis to slide over. Defaults to the last axis.
        copy : bool
            Return strided array as copy to avoid sideffects when manipulating the
            output array.
        Returns
        -------
        data : numpy array
            A matrix where row in last dimension consists of one instance
            of the sliding window.
        Notes
        -----
        - Be wary of setting `copy` to `False` as undesired sideffects with the
          output values may occurr.
        Examples
        --------
        """
        if stepsize is None:
            stepsize = size
        if axis >= data.ndim:
            raise ValueError(
                "Axis value out of range"
            )

        if stepsize < 1:
            raise ValueError(
                "Stepsize may not be zero or negative"
            )

        if size > data.shape[axis]:
            raise ValueError(
                "Sliding window size may not exceed size of selected axis"
            )

        shape = list(data.shape)
        shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
        shape.append(size)

        strides = list(data.strides)
        strides[axis] *= stepsize
        strides.append(data.strides[axis])

        strided = np.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides
        )

        if copy:
            return strided.copy()
        else:
            return strided




    def run_detector_on_ts(self, X, model):

        Y_pred = torch.zeros(X.size(0), X.size(2)).float()
        for i in range(0, Y_pred.shape[0], self.params['batch_size']):
            i_start = i
            i_stop = i + self.params['batch_size']
            if i_stop > Y_pred.shape[0]:
                i_stop = Y_pred.shape[0]
            X_test = X[i_start:i_stop]
            X_test = X_test.cuda(self.device)
            with torch.no_grad():
                out = model(X_test)
                Y_pred[i_start:i_stop,:] = out[:,0,:]

        return Y_pred


    def get_picks_from_ts(self,prob):

        picks = []
        pick_values = []
        trigs = trigger_onset(prob, self.params['min_proba'], 0.01)
        for trig in trigs:

            if trig[1] == trig[0]:
                continue

            trig_dur = (trig[1] - trig[0])

            if (trig_dur) < self.params['min_trig_dur']:
                continue

            pick = np.argmax(prob[trig[0]:trig[1]])+trig[0]
            picks.append(pick)

            pick_value = prob[pick]
            pick_values.append(pick_value)

        return picks, pick_values

    def get_onset(self,pick,pick_value):
        '''
            Creating a Gaussian to back propogate into the volume
        '''
        onset = np.zeros((self.onset.shape[-1]))
        for idpk,pk in enumerate(pick):
            idxmin = pk-len(np.where(self.GauX<0)[0])
            idxmax = (pk+1+len(np.where(self.GauX>0)[0]))
            onset[idxmin:idxmax] += self.GauY#*pick_value[idpk]
        return onset



    def compute_probs(self,ids):

        # Creating blank probability structures to fill
        prob_P = np.zeros((self.signal.shape[-1]))
        prob_S = np.zeros((self.signal.shape[-1]))

        # Determine if the length of the data is the same as the length of the number of features
        btch = round(self.signal.shape[-1] / self.n_feat)

        for ii in range(btch):
            lind = ii*self.n_feat
            hind = (ii+1)*self.n_feat


            if self.signal.shape[-1] <= self.n_feat:

                # Padding signal at the end with zeros
                fdsignal = self.signal[:,:,:]
                crsignal = np.zeros((self.signal.shape[0],self.signal.shape[1],self.n_feat))
                crsignal[:,:,:self.signal.shape[-1]] = fdsignal
                del fdsignal

                # Determining the probability on each of the batch sizes
                X = np.dstack([self.sliding_window(crsignal[idc,ids,:], self.n_feat) for idc in [0,1,2]])
                tr_max = np.max(np.abs(X), axis=(1,2))
                X /= tr_max[:,None,None]
                X = torch.from_numpy(X).float().permute(0, 2, 1)

                # Creating the P- and S-Probability functions
                tmpP = self.run_detector_on_ts(X,self.model_P).numpy().flatten()
                tmpS = self.run_detector_on_ts(X,self.model_S).numpy().flatten()

                prob_P = tmp[:self.signal.shape[-1]]
                prob_s = tmp[:self.signal.shape[-1]]
                continue

            elif hind <= self.signal.shape[-1]:
                # Determining the probability on each of the batch sizes
                X = np.dstack([self.sliding_window(self.signal[idc,ids,lind:hind], self.n_feat) for idc in [0,1,2]])
                tr_max = np.max(np.abs(X), axis=(1,2))
                X /= tr_max[:,None,None]
                X = torch.from_numpy(X).float().permute(0, 2, 1)

                # Creating the P- and S-Probability functions
                prob_P[lind:hind] = self.run_detector_on_ts(X,self.model_P).numpy().flatten()
                prob_S[lind:hind] = self.run_detector_on_ts(X,self.model_S).numpy().flatten()

            else:
                # Padding signal at the end with zeros
                fdsignal = self.signal[:,:,lind:]


                crsignal = np.zeros((self.signal.shape[0],self.signal.shape[1],self.n_feat))
                crsignal[:,:,:fdsignal.shape[-1]] = fdsignal
                del fdsignal

                # Determining the probability on each of the batch sizes
                X = np.dstack([self.sliding_window(crsignal[idc,ids,:], self.n_feat) for idc in [0,1,2]])
                tr_max = np.max(np.abs(X), axis=(1,2))
                X /= tr_max[:,None,None]
                X = torch.from_numpy(X).float().permute(0, 2, 1)

                # Creating the P- and S-Probability functions
                tmpP = self.run_detector_on_ts(X,self.model_P).numpy().flatten()
                tmpS = self.run_detector_on_ts(X,self.model_S).numpy().flatten()

                prob_P[lind:] = tmpP[:(self.signal.shape[-1]-lind)]
                prob_S[lind:] = tmpS[:(self.signal.shape[-1]-lind)]

        return prob_P,prob_S

    def compute_onset(self):
        c = 0 
        self.PICKS = {}
        for ids in range(self.signal.shape[1]):


            prob_P,prob_S = self.compute_probs(ids)



            # Creating the P- and S-wave picks for each station
            #p_picks, p_pick_value = self.get_picks_from_ts(prob_P)
            #s_picks, s_pick_value = self.get_picks_from_ts(prob_S)

            self.onset[0,ids,:] = prob_P#self.get_onset(p_picks,p_pick_value)
            self.onset[1,ids,:] = prob_S#self.get_onset(s_picks,s_pick_value)

            #self.PICKS['{}_P'.format(str(ids))] = p_picks
            #self.PICKS['{}_S'.format(str(ids))] = s_picks

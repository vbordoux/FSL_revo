
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import RandomOverSampler
import os
import librosa
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
from scipy.signal import butter, lfilter, freqz
from tqdm import tqdm
from torch import Tensor, from_numpy, cat
import matplotlib.pyplot as plt
from unit_test import test_filter_response_stability
from random import randint

import warnings
warnings.filterwarnings("ignore")


def class_to_int(label_array,class_set):

    '''  Convert class label to integer

    Args:
    -label_array: label array
    -class_set: unique classes in label_array

    Out:
    -y: label to index values
    '''
    label2indx = {label:index for index,label in enumerate(class_set)}
    y = np.array([label2indx[label] for label in label_array])
    return y



def balance_class_distribution(X,Y):

    '''  Class balancing through Random oversampling
    Args:
    -X: Feature
    -Y: labels

    Out:
    -X_new: Feature after oversampling
    -Y_new: Oversampled label list
    '''

    x_index = [[index] for index in range(len(X))]
    set_y = set(Y)


    ros = RandomOverSampler(random_state=42)
    x_unifm, y_unifm = ros.fit_resample(x_index, Y)
    unifm_index = [index_new[0] for index_new in x_unifm]

    X_new = np.array([X[index] for index in unifm_index])

    sampled_index = [idx[0] for idx in x_unifm]
    Y_new = np.array([Y[idx] for idx in sampled_index])

    return X_new,Y_new


def norm_params(X):

    '''  Normalize features
        Args:
        - X : Features

        Out:
        - mean : Mean of the feature set
        - std: Standard deviation of the feature set
        '''


    mean = np.mean(X)

    std = np.std(X)
    return mean, std



class Datagen(object):

    def __init__(self, conf):

        hdf_path = os.path.join(conf.path.feat_train, 'Mel_train.h5')
        hdf_train = h5py.File(hdf_path, 'r+')
        self.x = hdf_train['features'][:]
        self.labels = [s.decode() for s in hdf_train['labels'][:]]

        class_set = set(self.labels)

        self.y = class_to_int(self.labels,class_set)
        self.x,self.y = balance_class_distribution(self.x,self.y)
        array_train = np.arange(len(self.x))
        _,_,_,_,train_array,valid_array = train_test_split(self.x,self.y,array_train,random_state=42,stratify=self.y)
        self.train_index = train_array
        self.valid_index = valid_array
        self.mean,self.std = norm_params(self.x[train_array])


    def feature_scale(self,X):

        return (X-self.mean)/self.std



    def generate_train(self):

        ''' Returns normalized training and validation features.
        Args:
        -conf - Configuration object
        Out:
        - X_train: Training features
        - X_val: Validation features
        - Y_train: Training labels
        - Y_Val: Validation labels
        '''

        train_array = sorted(self.train_index)
        valid_array = sorted(self.valid_index)
        X_train = self.x[train_array]
        Y_train = self.y[train_array]
        X_val = self.x[valid_array]
        Y_val = self.y[valid_array]
        X_train = self.feature_scale(X_train)
        X_val = self.feature_scale(X_val)
        return X_train,Y_train,X_val,Y_val




class Datagen_test(Datagen):

    def __init__(self,hf,conf):
        super(Datagen_test, self).__init__(conf= conf)


        self.x_pos = hf['feat_pos'][:]
        self.x_neg = hf['feat_neg'][:]
        self.x_query = hf['feat_query'][:]
        self.hop_seg = hf['hop_seg'][:]
    

    def generate_eval(self):

        '''Returns normalizedtest features

        Output:
        - X_pos: Positive set features. Positive class prototypes will be calculated from this
        - X_query: Query set. Onset-offset prediction will be made on this set.
        - X_neg: The entire audio file. Will be used to calculate a negative prototype.
        '''

        X_pos = (self.x_pos)
        X_neg = (self.x_neg)
        X_query = (self.x_query)
        X_pos = self.feature_scale(X_pos)
        X_neg = self.feature_scale(X_neg)
        X_query = self.feature_scale(X_query)

        return X_pos, X_neg, X_query,self.hop_seg




# ---------------------------------------------------
# FUNCTION version of sliding windows that return a Dataframe 
# ---------------------------------------------------

class Datagen_test_wav(object):

    def __init__(self, conf):

        '''
        Get the annotation file, get the wave file
        For a given call type, extract the 5 first annotations rows
        Extract the 5 wav portion corresponding to the begin and end of the annotations (X_pos)
        Chunk all the file in segment of length conf.seg_len (X_neg)
        Copy X_neg and
        Remove all annotation before the end of the last 5th (X_query)
        return X_pos, X_neg and X_query
        '''

        self.sr = conf.features.sr
        self.n_shot = conf.train.n_shot
        self.seg_len = conf.features.seg_len
        self.call_type = conf.features.call_type
        self.fmin, self.fmax = conf.features.fmin, conf.features.fmax

        

    def generate_eval(self, audio_filepath=None, annot_filepath=None, apply_filter=False):
        '''Generate online dataset for one sound file and one annotation file
        Create pos, neg and query array to create dataloader for prototypes
            - audio_filepath: path to wav file
            - annot_filepath: path to txt file (annotation in Raven format)
        Output:
            - X_pos: Positive set features. Positive class prototypes will be calculated from this
            - X_query: Query set. Onset-offset prediction will be made on this    #     self.mean,self.std = norm_params(self.x[train_array])
        '''
        # Load annotation and wav file
        
        df_annot = pd.read_csv(annot_filepath, sep='\t')
        waveform, file_sr = torchaudio.load(audio_filepath, )
        if file_sr != 16000:
            transform = torchaudio.transforms.Resample(file_sr, 16000)
            waveform = transform(waveform)
            self.sr = 16000

        df_pos = df_annot[df_annot['Type'] == self.call_type]
        # If the given file do not contain the minimum number of annotation, skip the file
        if len(df_pos) < self.n_shot:
            return [], [], [], 0, 0, 0

        # Apply bandpass filter
        if apply_filter:
            order = 4 # in previous test if order is increased the filter is unstable
            cutoffs = [self.fmin, self.fmax]
            np_waveform = butter_bandpass_filter(waveform, cutoffs, self.sr, order)

            # Uncomment the line below to check the frequency response of the filter selected
            # test_filter_response_stability(waveform, order, cutoffs, self.sr)

            waveform = Tensor.float(from_numpy(np_waveform))
        
        # Normalize the waveform
        waveform = (waveform - waveform.mean())/waveform.std()

        # Create features for positive proto
        n_shot_df = df_pos.sort_values('Begin Time (s)').head(self.n_shot)

        X_pos = []
        pos_annot_bounds = []

        for i, row in n_shot_df.iterrows():
            start_wav = int(row['Begin Time (s)']*self.sr)
            end_wav = int(row['End Time (s)']*self.sr)

            # AVES minimal input is 25 ms, if smaller, make the segment 20 ms long
            if (end_wav - start_wav)/self.sr < 0.025:
                end_wav = int(start_wav + 0.025*self.sr)
            
            pos_annot_bounds.append((start_wav, end_wav))
            X_pos.append(waveform[0][start_wav:end_wav])

        # Compute pos proto in frame instead of windows
        seg_len_in_sample = int(self.seg_len * self.sr)
        X_pos_concat = cat(X_pos, 0)
        num_segments = int(len(X_pos_concat) // seg_len_in_sample)
        X_pos = np.array_split(X_pos_concat[:num_segments * seg_len_in_sample], num_segments) # Last part of the file (size<segment) is discarded

        # Remove the selection of the head above to compute the length of all pos file and compute density
        # print('Density in file : ', sum(len(pos_sample) for pos_sample in X_pos)/len(waveform[0]))

        # # Compute the mean length of X_pos samples to use to generate seg_len
        # mean_length = sum(len(sample) for sample in X_pos) / len(X_pos)
        # self.seg_len = mean_length / self.sr
        
        # Save the ending time of the last annotation (where to start query set)
        last_annot_endtime = int(n_shot_df.iloc[-1]['End Time (s)']*self.sr)
        self.start_query = last_annot_endtime

        # Compute the proto by averaging all the space between the pos_call
        X_neg = neg_proto_all_between_pos(pos_annot_bounds, waveform, seg_len_in_sample)
        if len(X_neg) < self.n_shot:
            print("WARNING: Not enough negative samples between pos call, go for whole file method unexpected behavior might happen")
            num_segments = int(len(waveform[0]) // seg_len_in_sample)
            X_neg = np.array_split(waveform[0][:num_segments * seg_len_in_sample], num_segments) # Last part of the file (size<segment) is discarded

        # Find the shortest element in X_pos list
        min_length = min(len(sample) for sample in X_pos)
        min_length_sec = min_length / self.sr


        # Create features for query set
        query_waveform = waveform[0][last_annot_endtime:]
        num_segments_query = len(query_waveform) // seg_len_in_sample
        X_query = np.array_split(query_waveform[:num_segments_query * seg_len_in_sample], num_segments_query)

        # print(f"X_pos shape {len(X_pos)}, X_neg shape {len(X_neg)}, X_query shape {len(X_query)}")
        self.x_pos = X_pos
        self.x_neg = X_neg
        self.x_query = X_query

        return X_pos, X_neg, X_query, self.start_query, min_length_sec, self.seg_len
        



def butter_bandpass(cutoffs, fs, order=5):
    return butter(order, cutoffs, fs=fs, btype='band', analog=False)

def butter_bandpass_filter(audio, cutoffs, fs, order=5):
    b, a = butter_bandpass(cutoffs, fs, order=order)
    y = lfilter(b, a, audio)
    return y



def neg_proto_all_between_pos(pos_annot_bounds, full_waveform, seg_len_in_sample):
    '''
    Function to convert all the space between positive annotations to negative samples

    Input:
    - waveform: the whole waveform
    - pos_annot_bounds: list of the start and end of the positive annotations

    Return:
    - X_neg: list of the negative samples
    '''
    # Select only the section of the waveform before the last positive sample (discard last annotation + query set)
    waveform = full_waveform.squeeze()[:pos_annot_bounds[-1][0]]

    # Create a new waveform with only the sections between the postitive annotations
    for bound in reversed(pos_annot_bounds[:-1]):
        waveform = cat((waveform[:bound[0]], waveform[bound[1]:]), 0)

    # Compute the negative sample of segment_length and discard the rest
    num_segments_query = len(waveform) // seg_len_in_sample
    X_neg = np.array_split(waveform[:num_segments_query * seg_len_in_sample], num_segments_query)

    return X_neg


# OLD STUFF / TRASH

# def neg_proto_sample_between_pos(n_shot, pos_annot_bounds, waveform, last_annot_endtime, mean_pos_length):
#     X_neg = []
#     # Create a list of negative sample from interval between 0 and the last positive annotation outside of positive annotation intervals
#     infinite_loop_check = 0
#     while len(X_neg) != n_shot and infinite_loop_check < 500:
#         flag = 0
#         # Draw randomly the start of an interval outside of the query set
#         start_idx = randint(0, len(waveform[0][0:last_annot_endtime-int(mean_pos_length)]))
#         end_idx = start_idx + int(mean_pos_length)
#         # if the random invertal overlap one of the positive annotation raise a flag that discard the sample
#         for bound in pos_annot_bounds:
#             if (bound[0] < start_idx < bound[1]) or (bound[0] < start_idx < bound[1]):
#                 flag = 1
#         if flag == 0:
#             X_neg.append(waveform[0][start_idx:end_idx])
#         infinite_loop_check += 1
#     return X_neg, infinite_loop_check
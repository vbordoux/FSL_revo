from tqdm import tqdm
import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from Model import AvesClassifier
import torchaudio
from Datagenerator import butter_bandpass_filter
import torch
from torch import Tensor, from_numpy
from glob import glob
import numpy as np



def plot_2d_embedding(result, label):
  '''
  Display a plot with the two first features of an embedding and display labels as colors
  param result: embedding
  param label: label
  '''
  result_df = pd.DataFrame({'component_1': result[:,0], 'component_2': result[:,1], 'label': label})
#   result_df.sort_values(by=['label'])
  fig, ax = plt.subplots(1)

  sns.scatterplot(x='component_1', y='component_2', hue='label', data=result_df, ax=ax,s=120)
  ax.set_aspect('equal')
  ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
  plt.show()



def tsne_embedding(X, n_components, perplexity = 10, n_iter = 1000, learning_rate='auto'):
  # Fit and transform X into a reduced space of n_components dimensions using T-SNE
  # https://distill.pub/2016/misread-tsne/?_ga=2.135835192.888864733.1531353600-1779571267.1531353600
  tsne = TSNE(n_components,perplexity=perplexity,n_iter=n_iter, method='exact', learning_rate=learning_rate)
  return tsne.fit_transform(X)



# Full dataset loader for TSNE
class Datagen_test_wav_for_tsne(object):

    def __init__(self, sr, seg_len, fmin, fmax):

        '''
        Get the annotation file, get the wave file
        For a given call type, extract the 5 first annotations rows
        Extract the 5 wav portion corresponding to the begin and end of the annotations (X_pos)
        Chunk all the file in segment of length conf.seg_len (X_neg)
        Copy X_neg and
        Remove all annotation before the end of the last 5th (X_query)
        return X_pos, X_neg and X_query
        '''

        self.sr = sr
        self.seg_len = seg_len
        self.fmin, self.fmax = fmin, fmax

        
    def generate_eval(self, audio_filepath=None, annot_filepath=None, call_list=['POS'], apply_filter=False):
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
        
        # Select only the call type in the list
        df_annot = df_annot[df_annot['Type'].isin(call_list)]

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


        X_pos = []
        Y_pos = []
        pos_annot_bounds = []

        for i, row in df_annot.iterrows():
            start_wav = int(row['Begin Time (s)']*self.sr)
            end_wav = int(row['End Time (s)']*self.sr)
            
            pos_annot_bounds.append((start_wav, end_wav))
            X_pos.append(waveform[0][start_wav:end_wav])
            Y_pos.append(row['Type'])

        
        # Compute the mean duration of positive sample
        mean_length = sum(len(sample) for sample in X_pos) / len(X_pos)

        # Draw randomly as many negative sample as positive of average positive sample duration
        X_neg = neg_proto_sample_between_pos(len(X_pos), pos_annot_bounds, waveform, mean_length)
        Y_neg = ['Noise']*len(X_neg)

        return X_pos, Y_pos, X_neg, Y_neg
    

def neg_proto_sample_between_pos(n_shot, pos_annot_bounds, waveform, mean_pos_length):
    X_neg = []
    # Create a list of negative sample from interval between 0 and the last positive annotation outside of positive annotation intervals
    while len(X_neg) != n_shot:
        flag = 0
        # Draw randomly the start of an interval outside of the query set
        start_idx = np.random.randint(0, len(waveform[0]))
        end_idx = start_idx + int(mean_pos_length)
        # if the random invertal overlap one of the positive annotation raise a flag that discard the sample
        for bound in pos_annot_bounds:
            if (bound[0] < start_idx < bound[1]) or (bound[0] < start_idx < bound[1]):
                flag = 1
        if flag == 0:
            X_neg.append(waveform[0][start_idx:end_idx])
    return X_neg



if __name__ == '__main__':

    device = 'cuda'
    model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
    model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
    emb_dim = 768
    sr=16000
    call_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    encoder = AvesClassifier(model_config_path, model_path, trainable=False, embedding_dim=emb_dim, sr=sr)
    encoder.to(device)
    encoder.eval()

    # Create waveform Dataloader if AVES
    gen_eval = Datagen_test_wav_for_tsne(sr=sr, seg_len=0.2, fmin=50, fmax=2000)

    wav_files_path = '/home/reindert/Valentin_REVO/DCASE_2022/GR_Set/GR'
    annot_files_path = '/home/reindert/Valentin_REVO/DCASE_2022/GR_Set'

    # TODO change glob to listdir
    all_wav_files = [file for file in glob(os.path.join(wav_files_path,'*.wav'))]
    all_annot_files = [file for file in glob(os.path.join(annot_files_path,'*.txt'))]
    all_wav_files.sort()
    all_annot_files.sort()
    
    # Use to run on all files
    feat_array = torch.Tensor().to(device)
    y_all_files = []

    for audio_file, annotation_file in zip(all_wav_files, all_annot_files):
        
        # Use to run per file
        feat_array = torch.Tensor().to(device)
        y_all_files = []

        # Get embedding from all positive sample
        X_pos, Y_pos, X_neg, Y_neg = gen_eval.generate_eval(audio_file, annotation_file, call_list, apply_filter=True)

        print('\n ---------------------------------------------------------------')
        print(f" File {os.path.basename(audio_file)}")
        print(' ---------------------------------------------------------------')
     
        # Add positive embeddings to the main array
        for sample in tqdm(X_pos):
            x_pos = Tensor(sample).unsqueeze(0)
            x_pos = x_pos.to(device)
            with torch.no_grad():
                feat_pos = encoder(x_pos)
            feat_array = torch.cat((feat_array, feat_pos), dim=0)
        # Add positive labels to the main array
        y_all_files += Y_pos

        # Add negative embeddings to the main array
        for sample in tqdm(X_neg):
            x_neg = Tensor(sample).unsqueeze(0)
            x_neg = x_neg.to(device)
            with torch.no_grad():
                feat_neg = encoder(x_neg)
            feat_array = torch.cat((feat_array, feat_neg), dim=0)
        # Add negative labels to the main array
        y_all_files += Y_neg

        # Remove indentation bellow to run on all files
        feat_array = feat_array.cpu()

        # Compute 2D embedding with T-SNE and display it
        tsne_X_2d = tsne_embedding(feat_array, n_components=2, perplexity=25, n_iter=500)
        plot_2d_embedding(tsne_X_2d, y_all_files)




# Compute the embedding of the sample with AVES based on 3s windows

# ---------------------------------------------------
# FUNCTION version of sliding windows that return a Dataframe 
# # ---------------------------------------------------
# import librosa

# def sliding_window_cuting(audio_filepath, annot_filepath, wind_dur=1., coverage_threshold=0.2):
#   '''
#   Generate chunks of audio with label associated based on annotation file and audio file
#   '''

#   df_annot = pd.read_csv(annot_filepath, sep='\t')
#   waveform, sr = librosa.load(audio_filepath, sr=48000)

#   # Calculate frame length in samples
#   frame_length_sample = int(wind_dur * sr)

#   # Create a list to store chunks and start times
#   chunks = []
#   start_time = []

#   # Iterate through the audio waveform and create chunks
#   for i in range(0, len(waveform) - frame_length_sample + 1, frame_length_sample):
#       chunk = waveform[i:i+frame_length_sample]
#       chunks.append(chunk)
#       start_time.append(i / sr)  # Convert sample index to time in seconds

#   df_chunks = pd.DataFrame({'Audio': chunks, 'Start_time': start_time})

#   # Create label if the frame contains annotation
#   labels = []

#   for i, chunk in tqdm(df_chunks.iterrows()):
#       start_window = chunk['Start_time']
#       end_window = start_window + wind_dur

#       # Check if there are annotations in the current window
#       begin_anot_presence = (start_window < df_annot['Begin Time (s)']) & (df_annot['Begin Time (s)'] < end_window)
#       end_anot_presence = (start_window < df_annot['End Time (s)']) & (df_annot['End Time (s)']< end_window)
#       full_anot_presence = ((df_annot['Begin Time (s)'] < start_window) & (start_window < df_annot['End Time (s)'])) \
#                               &((df_annot['Begin Time (s)'] < end_window) & (end_window < df_annot['End Time (s)']))

#       # Extract the annotations that overlap with the chunk
#       annotations_in_window = df_annot[(begin_anot_presence) | (end_anot_presence) | (full_anot_presence)]

#       # Calculate total annotation duration within the window

#       annotations_in_window['Duration'] = annotations_in_window.apply(
#           lambda row: min(end_window, row['End Time (s)']) - max(start_window, row['Begin Time (s)']),
#           axis=1
#       )
      
#       # annotations_duration = annotations_in_window.apply(
#       #     lambda row: min(end_window, row['End Time (s)']) - max(start_window, row['Begin Time (s)']),
#       #     axis=1
#       # )

#       total_annotation_duration = annotations_in_window['Duration'].sum()
#       # Select type where duration is maximum
#       if len(annotations_in_window)>0:
#         longest_annotation = annotations_in_window.loc[annotations_in_window['Duration'].idxmax()]['Type']

#       # If more than the coverage threshold of the annotation is within the window, assign a positive label
#       label = longest_annotation if total_annotation_duration > coverage_threshold * wind_dur else 'Noise'
#       labels.append(label)

#   df_chunks['Label'] = labels

#   print('Number of samples in each of the class: ')
#   print(df_chunks['Label'].value_counts())

#   return df_chunks



# if __name__ == '__main__':

#     device = 'cuda'
#     model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
#     model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
#     emb_dim = 768
#     sr=16000
#     call_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

#     encoder = AvesClassifier(model_config_path, model_path, trainable=False, embedding_dim=emb_dim, sr=sr)
#     encoder.to(device)
#     encoder.eval()

#     # Load audio file and resample to 48000Hz
#     filepath = '/home/reindert/Valentin_REVO/DCASE_2022/GR_Set/GR/SanctSound_GR03_05_5421_201005023324part1.wav'
#     annot_path = '/home/reindert/Valentin_REVO/DCASE_2022/GR_Set/SanctSound_GR03_05_5421_201005023324part1.Table.1.selections.txt'

#     sig, rate = librosa.load(filepath, sr=48000)

#     df_chunks = sliding_window_cuting(filepath, annot_path, wind_dur = 3., coverage_threshold = 0)
    
#     # Resampling ta AVES native freq
#     df_chunks['Audio'] = df_chunks['Audio'].apply(lambda x: librosa.resample(y = x, orig_sr = 48000, target_sr = 16000))

#     chunks = [row['Audio'] for _idx, row in df_chunks.iterrows()]
#     labels = [row['Label'] for _idx, row in df_chunks.iterrows()]

#     feat_array = torch.Tensor().to(device)

    
#     # Add positive embeddings to the main array
#     for sample in tqdm(chunks):
#         x_pos = Tensor(sample).unsqueeze(0)
#         x_pos = x_pos.to(device)
#         with torch.no_grad():
#             feat_pos = encoder(x_pos)
#         feat_array = torch.cat((feat_array, feat_pos), dim=0)
            
#     # Display TSNE
#     # labels = np.zeros(embeddings.shape[0])
#     tsne_X_2d = tsne_embedding(feat_array.cpu(), n_components=2, perplexity=10, n_iter=1000)
#     plot_2d_embedding(tsne_X_2d, labels)
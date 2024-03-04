import torch
from torch.nn import functional as F
from Model import AvesClassifier, ResNet
from Datagenerator import Datagen_test, Datagen_test_wav
import numpy as np
from batch_sampler import EpisodicBatchSampler
from tqdm import tqdm
import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def init_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    
    n, m = x.size(0), y.size(0)
    d = x.size(1)

    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)



def get_probability(proto_pos,neg_proto,query_set_out):
    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set

     Out:
     - Probabiility array for the positive class
     """
    
    prototypes = torch.stack([proto_pos,neg_proto]).squeeze(1)
    dists = euclidean_dist(query_set_out,prototypes)

    '''  Taking inverse distance for converting distance to probabilities'''
    logits = -dists

    #Testing prototypes similarity
    dist_proto = euclidean_dist(proto_pos.unsqueeze(0), neg_proto.unsqueeze(0))
    # print("Similarity between pos and neg proto is ", 1/(1+dist_proto))
    # print("Where 0 is very different and 1 very similar")

    prob = torch.softmax(logits,dim=1)
    inverse_dist = torch.div(1.0, dists)
    
    #prob = torch.softmax(inverse_dist,dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,0]

    return prob_pos.detach().cpu().tolist()



def evaluate_prototypes_AVES(conf=None, encoder = None, device= None, audio_file = None, annotation_file = None):
    '''
    This function is ugly for now but manage to generate positive, negative and query prototypes with AVES
    It returns two arrays onset and offset, respectively the start and the end of predicted events
    '''

    # Create waveform Dataloader if AVES
    gen_eval = Datagen_test_wav(conf = conf)

    X_pos, X_neg, X_query, start_query, min_length, seg_len = gen_eval.generate_eval(audio_file, annotation_file, apply_filter=conf.eval.apply_filter)

    'List for storing the combined probability across all iterations'
    prob_comb = [] 

    print('\n ---------------------------------------------------------------')
    print(f" File {os.path.basename(audio_file)}")
    print(' ---------------------------------------------------------------')

    # if len(X_pos) < conf.train.n_shot:
    #     print("Not enough positive samples, file is skipped")
    #     return [], [], 3600
   
    # If the file have enough annotation, process to evaluation
    X_pos = torch.tensor(torch.stack(X_pos, dim=0))
    pos_dataset = torch.utils.data.TensorDataset(X_pos)
    pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=None,batch_size=conf.eval.pos_batch_size,shuffle=False)

    # If the file have enough annotation, process to evaluation
    X_neg = torch.tensor(torch.stack(X_neg, dim=0))
    neg_dataset = torch.utils.data.TensorDataset(X_neg)
    neg_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None,batch_size=conf.eval.negative_set_batch_size,shuffle=False)

    X_query = torch.tensor(torch.stack(X_query, dim=0))
    query_dataset = torch.utils.data.TensorDataset(X_query)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=conf.eval.query_batch_size,shuffle=False)


    # print("\n Adaptative segment length: Average length of positive samples: ", seg_len)
    print(f"Creating positive prototype from {len(X_pos)} samples")

    pos_iterator = iter(pos_loader)
    feat_array_pos = torch.Tensor().to(device)

    for batch in tqdm(pos_iterator):
        x_pos = batch[0]
        x_pos = x_pos.to(device)
        feat_pos = encoder(x_pos)
        feat_array_pos = torch.cat((feat_array_pos, feat_pos), dim=0)
                            
    # Compute positive prototype as the mean of all positive embeddings
    pos_proto = feat_array_pos.mean(dim=0).to(device)

    # for pos_sample in tqdm(X_pos):
    #     feat = encoder(pos_sample.unsqueeze(0).cuda())
    #     feat = feat.cpu()
    #     feat_mean = feat.mean(dim=0).unsqueeze(0)
    #     pos_set_feat = torch.cat((pos_set_feat, feat_mean), dim=0)
    # pos_proto = pos_set_feat.mean(dim=0)


    prob_pos_iter = []
    
    print(f"Creating negative prototype from {len(X_neg)} samples")

    neg_iterator = iter(neg_loader)
    feat_array_neg = torch.Tensor().to(device)

    for batch in tqdm(neg_iterator):
        x_neg = batch[0]
        x_neg = x_neg.to(device)
        feat_neg = encoder(x_neg)
        feat_array_neg = torch.cat((feat_array_neg, feat_neg), dim=0)
                            
    # Compute negative prototype as the mean of all negative embeddings
    proto_neg = feat_array_neg.mean(dim=0).to(device)

    # Create query set
    print("Evaluating query set with prototypes")
    q_iterator = iter(q_loader)
    for batch in tqdm(q_iterator):
        x_q = batch[0]
        x_q = x_q.to(device)
        x_query = encoder(x_q)

        pos_proto = pos_proto.detach().cpu()
        proto_neg = proto_neg.detach().cpu()
        x_query = x_query.detach().cpu()
        
        probability_pos = get_probability(pos_proto, proto_neg, x_query)
        prob_pos_iter.extend(probability_pos)

    prob_comb.append(prob_pos_iter)
    prob_final = np.mean(np.array(prob_comb),axis=0)
    
    thresh = conf.eval.threshold
    
    krn = np.array([1, -1])
    prob_thresh = np.where(prob_final > thresh, 1, 0)

    # prob_pos_final = prob_final * prob_thresh
    
    changes = np.convolve(krn, prob_thresh)

    # onset = start of events, offset = end of events
    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    onset = onset_frames * seg_len
    onset = onset + start_query / conf.features.sr

    offset = offset_frames * seg_len
    offset = offset + start_query / conf.features.sr
    assert len(onset) == len(offset)
    return onset, offset, min_length


# Function to evaluate from features (offline) with Resnet, not used atm
def evaluate_prototypes(conf=None,hdf_eval=None,device= None,strt_index_query=None):
    pass
#     """ Run the evaluation
#     Args:
#      - conf: config object
#      - hdf_eval: Features from the audio file
#      - device:  cuda/cpu
#      - str_index_query : start frame of the query set w.r.t to the original file

#      Out:
#      - onset: Onset array predicted by the model
#      - offset: Offset array predicted by the model
#       """
#     # Choose the model
#     # ----------------------------------
#     if conf.train.encoder == 'Resnet':
#         encoder = ResNet()
#         if device == 'cpu':
#             state_dict = torch.load(conf.path.best_model,map_location=torch.device('cpu'))
#             encoder.load_state_dict(state_dict['encoder'])
            
#         else:
#             state_dict = torch.load(conf.path.best_model)
#             encoder.load_state_dict(state_dict['encoder'])

        
#     encoder.to(device)
#     encoder.eval()

#     # Create spectrogram Dataloader if Resnet
#     # ------------------------------------------------------
#     gen_eval = Datagen_test(hdf_eval,conf)
#     X_pos, X_neg,X_query,hop_seg = gen_eval.generate_eval()
#     X_pos = torch.tensor(X_pos)
#     Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
#     X_neg = torch.tensor(X_neg)
#     Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
#     X_query = torch.tensor(X_query)
#     Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))
    
#     num_batch_query = len(Y_query) // conf.eval.query_batch_size
    
#     query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
#     q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=conf.eval.query_batch_size,shuffle=False)
#     query_set_feat = torch.zeros(0,48).cpu()
#     batch_samplr_pos = EpisodicBatchSampler(Y_pos, 2, 1, conf.train.n_shot)
#     pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
#     pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=None)

    
#     'List for storing the combined probability across all iterations'
#     prob_comb = [] 
#     emb_dim = 512
#     pos_set_feat = torch.zeros(0,emb_dim).cpu()

#     print("Creating positive prototype")
#     for batch in tqdm(pos_loader):
#         x,y = batch    
#         feat = encoder(x.cuda())
#         feat = feat.cpu()
#         feat_mean = feat.mean(dim=0).unsqueeze(0)
#         pos_set_feat = torch.cat((pos_set_feat, feat_mean), dim=0)
#     pos_proto = pos_set_feat.mean(dim=0)


#     iterations = conf.eval.iterations
#     for i in range(iterations):
#         prob_pos_iter = []
#         neg_indices = torch.randperm(len(X_neg))[:conf.eval.samples_neg]
#         X_neg = X_neg[neg_indices]
#         Y_neg = Y_neg[neg_indices]
        
#         feat_neg = encoder(X_neg.cuda())
#         feat_neg = feat_neg.detach().cpu()
#         proto_neg = feat_neg.mean(dim=0).to(device)
#         q_iterator = iter(q_loader)

#         print("Iteration number {}".format(i))

        

#         for batch in tqdm(q_iterator):
#             x_q, y_q = batch
#             x_q = x_q.to(device)
#             x_query = encoder(x_q)
            
#             proto_neg = proto_neg.detach().cpu()
#             x_query = x_query.detach().cpu()
            
#             probability_pos = get_probability(pos_proto, proto_neg, x_query)
#             prob_pos_iter.extend(probability_pos)

#         prob_comb.append(prob_pos_iter)
#     prob_final = np.mean(np.array(prob_comb),axis=0)
    
#     thresh = conf.eval.threshold
    
#     krn = np.array([1, -1])
#     prob_thresh = np.where(prob_final > thresh, 1, 0)

#     prob_pos_final = prob_final * prob_thresh
    
#     changes = np.convolve(krn, prob_thresh)

#     # onset = start of events, offset = end of events
#     onset_frames = np.where(changes == 1)[0]
#     offset_frames = np.where(changes == -1)[0]

#     str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr

#     onset = (onset_frames ) * (hop_seg) * conf.features.hop_mel / conf.features.sr
#     onset = onset + str_time_query

#     offset = (offset_frames ) * (hop_seg) * conf.features.hop_mel / conf.features.sr
#     offset = offset + str_time_query
#     assert len(onset) == len(offset)
#     return onset, offset



def raven_to_annot_csv(raven_filepath, dest_dir, annotation_column_name='Type', call_type=None):
    """ Transform a raven file into a format compatible with Proto
    Args:Returns normalized test features
     - eval_filepath: path to the file containing the annotations

     Out:
     - Create a csv file with annotations
    """
    raven_df = pd.read_csv(raven_filepath, sep='\t')
    annotation_df = pd.DataFrame({'Starttime':raven_df['Begin Time (s)'], 'Endtime':raven_df['End Time (s)'], 'Q':raven_df[annotation_column_name]})

    source_dir, raven_filename = os.path.split(raven_filepath)
    # Remove the calltype from the filename to generate the dictionnary key properly
    if os.path.splitext(raven_filename)[0].endswith(call_type):
        raven_filename = raven_filename[:-5] + raven_filename[-4:]

    if os.path.basename(raven_filename).endswith('Table.1.selections.txt'):
        raven_filename = raven_filename.replace('Table.1.selections.txt','txt')
    
    wav_filename = os.path.splitext(raven_filename)[0]+'.wav'
    csv_filename = os.path.splitext(raven_filename)[0]+'.csv'

    annotation_df['Audiofilename'] = wav_filename
    annotation_df = annotation_df[['Audiofilename','Starttime','Endtime', 'Q']]

    try:
        annotation_df.to_csv(os.path.join(dest_dir,csv_filename), sep=',', index=False)
        print("File saved at: ", os.path.join(dest_dir,csv_filename))
    except Exception as e:
         print("Could not save the annotations in a csv file:", e)
    

    
def eval_to_Raven(conf, eval_filepath):
    """ Transform a prediction file from Proto into a format compatible with Raven
    Args:
     - eval_filepath: path to the file containing the predictions

     Out:
     - Create one annotation file per filename in the prediction file that can be read by Raven
      """
    
    pred_csv = pd.read_csv(eval_filepath, dtype=str)

    PRED_FILE_HEADER = ["Audiofilename","Starttime","Endtime"]
    POS_VALUE = conf.features.call_type
    UNK_VALUE = 'UNK'

    #verify headers:
    if not ("Audiofilename" in list(pred_csv.columns) and "Starttime" in list(pred_csv.columns) and "Endtime" in list(pred_csv.columns)):
        print('Please correct the header of the prediction file. This should be', PRED_FILE_HEADER)
        exit(1)
    #  parse prediction csv
    #  split file into lists of events for the same audiofile.
    pred_events_by_audiofile = dict(tuple(pred_csv.groupby('Audiofilename')))

    predicted_events = pd.read_csv(eval_filepath)
    list_predicted_files = predicted_events.groupby(['Audiofilename'])

    source_dir, _filename = os.path.split(eval_filepath)
    raven_dir = os.path.join(source_dir, 'Raven files/')
    if not os.path.exists(raven_dir):
        os.makedirs(raven_dir)

    for _idx, predicted_file  in list_predicted_files:
        raven_file_df = pd.DataFrame({'Begin Time (s)': predicted_file['Starttime'], 'End Time (s)': predicted_file['Endtime']})
        raven_file_df.index.name = 'Selection'
        raven_file_df['View'] = 'Spectrogram 1'
        raven_file_df['Channel'] = 1
        raven_file_df['Low Freq (Hz)'] = conf.features.fmin
        raven_file_df['High Freq (Hz)'] = conf.features.fmax
        if "Q" in list(pred_csv.columns):
            raven_file_df['Type'] = predicted_file['Q']
        else:
            raven_file_df['Type'] = POS_VALUE

        if raven_file_df.index[0] == 0: # Raven does not support index starting at 0
            raven_file_df.index += 1

        dest_dir = raven_dir + os.path.splitext(predicted_file['Audiofilename'].iloc[0])[0]+'.txt'
        try:
            raven_file_df.to_csv(dest_dir, sep='\t')
            print('Raven annotation saved at:', dest_dir)
        except Exception as e:
            print("Could not save the annotations in a csv file:", e)



def select_calltype_raven_annot(conf, sourcefile_path, dest_dir):
    '''
    Save a copy of the raven file containing only the annotation with the label Call_type
    '''
    name, ext = os.path.splitext(os.path.basename(sourcefile_path))
    dest_file = name + conf['features']['call_type'] + ext

    df = pd.read_csv(sourcefile_path, sep='\t')
    df_G=df[df['Type']==conf['features']['call_type']]
    df_G.to_csv(dest_dir+dest_file, sep='\t')





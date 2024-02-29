import pandas as pd
import os
from glob import glob
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from torchaudio.models import wav2vec2_model
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split


class AvesClassifier(nn.Module):
    """ Uses AVES Hubert to embed sounds and classify """
    def __init__(self, config_path, model_path, n_classes, trainable, embedding_dim=768):
        super().__init__()
        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        # Freeze the AVES network
        self.trainable = trainable
        freeze_embedding_weights(self.model, trainable)
        # We will only train the classifier head
        self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=n_classes)
        self.audio_sr = 16000

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)
        return obj

    def forward(self, sig):
        """
        Input
          sig (Tensor): (batch, time)
        Returns
          mean_embedding (Tensor): (batch, output_dim)
          logits (Tensor): (batch, n_classes)
        """
        # extract_feature in the sorchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]
        mean_embedding = out.mean(dim=1) #over time
        logits = self.classifier_head(mean_embedding)
        return mean_embedding, logits
    
# Code to use while initially setting up the model
def freeze_embedding_weights(model, trainable):
  """ Freeze weights in AVES embeddings for classification """
  # The convolutional layers should never be trainable
  model.feature_extractor.requires_grad_(False)
  model.feature_extractor.eval()
  # The transformers are optionally trainable
  for param in model.encoder.parameters():
    param.requires_grad = trainable
  if not trainable:
    # We also set layers without params (like dropout) to eval mode, so they do not change
    model.encoder.eval()



# Code to use during training loop, to switch between eval and train mode
def set_eval_aves(model):
  """ Set AVES-based classifier to eval mode. Takes into account whether we are training transformers """
  model.classifier_head.eval()
  model.model.encoder.eval()



def set_train_aves(model):
  """ Set AVES-based classifier to train mode. Takes into account whether we are training transformers """
  # Always train the classifier head
  model.classifier_head.train()
  # Optionally train the transformer of the model
  if model.trainable:
      model.model.encoder.train()



def pad_to_duration(x, sr, duration_sec):
    """ Pad or clip x to a given duration """
    assert len(x.size()) == 1
    x_duration = x.size(0) / float(sr)
    max_samples = int(sr * duration_sec)
    if x_duration == duration_sec:
        return x
    elif x_duration < duration_sec:
        x = F.pad(x , (0, max_samples - x.size(0)), mode='constant')
        return x
    else:
        return x[:max_samples]




def train_one_epoch(model, dataloader, optimizer, loss_function):
    """ Update model based on supervised classification task """

    set_train_aves(model)
    loss_function = nn.CrossEntropyLoss()

    epoch_losses = []
    iterator = tqdm(dataloader)
    for i, batch_dict in enumerate(iterator):
        optimizer.zero_grad()
        if torch.cuda.is_available():
          batch_dict[0] = batch_dict[0].cuda()
          batch_dict[1] = batch_dict[1].cuda()

        embedding, logits = model(batch_dict[0])
        loss = loss_function(logits, batch_dict[1].to(torch.long))

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        if len(epoch_losses) > 10:
          iterator.set_description(f"Train loss: {np.mean(epoch_losses[-10:]):.3f}")

    return epoch_losses



def test_one_epoch(model, dataloader, loss_function, epoch_idx):
    """ Obtain loss and F1 scores on test set """

    set_eval_aves(model)

    # Obtain predictions
    all_losses = []
    all_predictions = []
    with torch.no_grad():
        for i, batch_dict in enumerate(dataloader):
            if torch.cuda.is_available():
                batch_dict[0] = batch_dict[0].cuda()
                batch_dict[1] = batch_dict[1].cuda()
            embedding, logits = model(batch_dict[0])
            all_losses.append(loss_function(logits, batch_dict[1].to(torch.long)))
            all_predictions.append(logits.argmax(1))

    # Format predictions and annotations
    all_losses = torch.stack(all_losses)
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_annotations = dataloader.dataset[:][1].numpy()
    #   all_annotations = dataloader.dataset.dataset_info[dataloader.dataset.annotation_name + "_int"].to_numpy() # since dataloader shuffle = False
    
    # Get confusion matrix
    cm = confusion_matrix(all_annotations, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    if epoch_idx == 19:
        disp.plot()
        disp.ax_.set_title(f"Test epoch {epoch_idx}")
    # Compute F1
    f1_scores = f1_score(all_annotations, all_predictions, average=None)
    macro_average_f1 = f1_score(all_annotations, all_predictions, average="macro")
    # Report
    print(f"Mean test loss: {all_losses.mean():.3f}, Macro-average F1: {macro_average_f1:.3f}")
    print("F1 by class:")
    print({k: np.round(s,decimals=4) for (k,s) in zip([0, 1], f1_scores)})
    return



def run(
      train_dataloader,
      test_dataloader,
      model_path,
      model_config_path,
      learning_rate,
      n_epochs,
      n_class
      ):


  print("Setting up model")
  model = AvesClassifier(model_config_path, model_path, n_class, False)
  if torch.cuda.is_available():
    model.cuda()

  print("Setting up optimizers")
  optimizer = torch.optim.Adam(model.classifier_head.parameters(), lr=learning_rate)

  print("Setting up loss function")
  loss_function = nn.CrossEntropyLoss()

  for epoch_idx in range(n_epochs):
    print(f"~~ Training epoch {epoch_idx}")
    train_one_epoch(model, train_dataloader, optimizer, loss_function)
    print(f"~~ Testing epoch {epoch_idx}")
    test_one_epoch(model, test_dataloader, loss_function, epoch_idx)

  return



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

    wav_files_path = '/home/reindert/Valentin_REVO/DCASE_2022/GR_Set/GR'
    annot_files_path = '/home/reindert/Valentin_REVO/DCASE_2022/GR_Set'

    # TODO change glob to listdir
    all_wav_files = [file for file in glob(os.path.join(wav_files_path,'*.wav'))]
    all_annot_files = [file for file in glob(os.path.join(annot_files_path,'*.txt'))]
    all_wav_files.sort()
    all_annot_files.sort()
    
    # Use to run on all files
    X_all = []
    Y_all = []
    X_test = []
    Y_test = []
    call_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    index_for_eval = [1,3]

    for audio_file, annotation_file, file_idx in zip(all_wav_files, all_annot_files, range(len(all_wav_files))):
        
        print(f"Processing file: {os.path.basename(audio_file)} ")
        if file_idx  in index_for_eval:
            print("File used for evaluation")
        else:
            print("File used for training")
        
        # Load annotation and wav file
        df_annot = pd.read_csv(annotation_file, sep='\t')
        waveform, file_sr = torchaudio.load(audio_file, )
        if file_sr != 16000:
            transform = torchaudio.transforms.Resample(file_sr, 16000)
            waveform = transform(waveform)
            sr = 16000
        
        # Select only the call type in the list
        # df_annot = df_annot[df_annot['Type'].isin(call_list)]

        # Apply bandpass filter
        # if apply_filter:
        #     order = 4 # in previous test if order is increased the filter is unstable
        #     cutoffs = [self.fmin, self.fmax]
        #     np_waveform = butter_bandpass_filter(waveform, cutoffs, self.sr, order)

            # Uncomment the line below to check the frequency response of the filter selected
            # test_filter_response_stability(waveform, order, cutoffs, self.sr)

            # waveform = Tensor.float(from_numpy(np_waveform))
        
        # Normalize the waveform
        waveform = (waveform - waveform.mean())/waveform.std()

        X_pos = []
        Y_pos = []
        pos_annot_bounds = []

        for i, row in df_annot.iterrows():
            start_wav = int(row['Begin Time (s)']*sr)
            end_wav = int(row['End Time (s)']*sr)
            
            pos_annot_bounds.append((start_wav, end_wav))
            X_pos.append(waveform[0][start_wav:end_wav])
            # Y_pos.append(row['Type'])
            Y_pos.append(1)

        
        # Compute the mean duration of positive sample
        mean_length = sum(len(sample) for sample in X_pos) / len(X_pos)

        # Draw randomly as many negative sample as positive of average positive sample duration
        X_neg = neg_proto_sample_between_pos(len(X_pos), pos_annot_bounds, waveform, mean_length)
        Y_neg = [0]*len(X_neg)
        filename = [audio_file]*(len(X_pos)+len(X_neg))

        # Extract samples for each files
        if file_idx in index_for_eval:
            X_test += X_pos + X_neg
            Y_test += Y_pos + Y_neg
        else:
            X_all += X_pos + X_neg
            Y_all += Y_pos + Y_neg

    # If the file have enough annotation, process to evaluation
    # pad or clip X samples to the same duration
    for idx, x in zip(range(len(X_all)), X_all):
        X_all[idx] = pad_to_duration(x, 16000, 0.5)

    for idx, x in zip(range(len(X_test)), X_test):
        X_test[idx] = pad_to_duration(x, 16000, 0.5)

    # Convert the list of array into a multidimentional tensor
    X_all = torch.stack(X_all)
    Y_all = torch.tensor(Y_all)

    # n_fft = 512
    # fig, axs = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(12,8))
    # ax_idx = 0

    # for datapoint_idx in np.random.randint(0, len(X_all), size=8):
    #     ax = axs.flatten()[ax_idx]
    #     datapoint = X_all[datapoint_idx]
    #     D = librosa.stft(datapoint.numpy(), n_fft=n_fft)
    #     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    #     librosa.display.specshow(S_db, y_axis='linear', sr=16000,
    #                                 x_axis='time', ax=ax, n_fft=n_fft)
    #     ax.set_title(Y_all[datapoint_idx].item())
    #     ax_idx+=1
    # plt.tight_layout()
    # plt.show()

    # x_train, x_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.4, random_state=11)

    x_train = X_all
    y_train = Y_all
    x_test = torch.stack(X_test)
    y_test = torch.tensor(Y_test)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=None,batch_size=8,shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_sampler=None,batch_size=8,shuffle=False)

    learning_rate = 0.001
    n_epochs = 20
    model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
    model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"

    run(
      train_loader,
      test_loader,
      model_path,
      model_config_path,
      learning_rate,
      n_epochs,
      n_class=2
    )
    
    plt.show()
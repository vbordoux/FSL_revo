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

from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, loss_function):
    """ Update model based on supervised classification task """

    set_train_aves(model)
    loss_function = nn.CrossEntropyLoss()

    epoch_losses = []
    iterator = tqdm(dataloader)
    for i, batch_dict in enumerate(iterator):
        optimizer.zero_grad()
        if torch.cuda.is_available():
          batch_dict["x"] = batch_dict["x"].cuda()
          batch_dict[dataloader.dataset.annotation_name] = batch_dict[dataloader.dataset.annotation_name].cuda()

        embedding, logits = model(batch_dict["x"])
        loss = loss_function(logits, batch_dict[dataloader.dataset.annotation_name].to(torch.long))

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        if len(epoch_losses) > 10:
          iterator.set_description(f"Train loss: {np.mean(epoch_losses[-10:]):.3f}")

    return epoch_losses

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

def test_one_epoch(model, dataloader, loss_function, epoch_idx):
  """ Obtain loss and F1 scores on test set """

  set_eval_aves(model)

  # Obtain predictions
  all_losses = []
  all_predictions = []
  with torch.no_grad():
    for i, batch_dict in enumerate(dataloader):
        if torch.cuda.is_available():
          batch_dict["x"] = batch_dict["x"].cuda()
          batch_dict[dataloader.dataset.annotation_name] = batch_dict[dataloader.dataset.annotation_name].cuda()
        embedding, logits = model(batch_dict["x"])
        all_losses.append(loss_function(logits, batch_dict[dataloader.dataset.annotation_name].to(torch.long)))
        all_predictions.append(logits.argmax(1))

  # Format predictions and annotations
  all_losses = torch.stack(all_losses)
  all_predictions = torch.cat(all_predictions).cpu().numpy()
  all_annotations = dataloader.dataset.dataset_info[dataloader.dataset.annotation_name + "_int"].to_numpy() # since dataloader shuffle = False
  # Get confusion matrix
  cm = confusion_matrix(all_annotations, all_predictions)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataloader.dataset.classes)
  disp.plot()
  disp.ax_.set_title(f"Test epoch {epoch_idx}")
  # Compute F1
  f1_scores = f1_score(all_annotations, all_predictions, average=None)
  macro_average_f1 = f1_score(all_annotations, all_predictions, average="macro")
  # Report
  print(f"Mean test loss: {all_losses.mean():.3f}, Macro-average F1: {macro_average_f1:.3f}")
  print("F1 by class:")
  print({k: np.round(s,decimals=4) for (k,s) in zip(dataloader.dataset.classes, f1_scores)})
  return

def run(
      dataset_dataframe,
      model_path,
      model_config_path,
      duration_sec,
      annotation_name,
      learning_rate,
      batch_size,
      n_epochs,
      aves_sr = 16000
      ):

  print("Setting up dataloaders")
  train_dataloader = get_dataloader(dataset_dataframe, True, aves_sr, duration_sec, annotation_name, batch_size)
  test_dataloader = get_dataloader(dataset_dataframe, False, aves_sr, duration_sec, annotation_name, batch_size)

  print("Setting up model")
  model = AvesClassifier(model_config_path, model_path, len(train_dataloader.dataset.classes), False)
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


run(
      dataset_dataframe=df,
      model_path="/content/aves-base-bio.torchaudio.pt",
      model_config_path="/content/aves-base-bio.torchaudio.model_config.json",
      duration_sec=1.0,
      annotation_name="call_type",
      learning_rate=1e-3,
      batch_size=20,
      n_epochs=10
    )
import argparse
import glob
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torchaudio
from torchaudio.sox_effects import apply_effects_file
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
import IPython


class SpeakerLevelDataset(Dataset):
    def __init__(self, base_path, splits, choices, model):
        self.speakers = self._getspeakerlist(base_path, splits, choices)
        self.model = model
    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        speaker_feature = []
        for chapter in glob.glob(os.path.join(self.speakers[idx], "*")):
            for feature_path in glob.glob(os.path.join(chapter, f"{self.model}-*")):
                feature = torch.load(feature_path).detach().cpu()
                feature = feature.squeeze()
                speaker_feature.append(np.array(feature).mean(axis=0))
        # print(len(speaker_feature))
        return speaker_feature, self.speakers[idx]
        
    def collate_fn(self, samples):
        return zip(*samples)

    def _getspeakerlist(self, base_path, splits, choices):
        speaker_list = []

        split_pathes = [os.path.join(base_path, split) for split in splits]

        split_choices = [(math.ceil(choices*(i+1)/len(split_pathes))-math.ceil(choices*i/len(split_pathes))) for i in range(len(split_pathes))]
        for split_path, split_choice in zip(split_pathes, split_choices):
            all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
            analyze_speakers = random.sample(all_speakers, k=min(split_choice, len(all_speakers)))
            
            speaker_list += analyze_speakers
            
        # print(len(speaker_list))
        return speaker_list


class UtteranceLevelDataset(Dataset):
    def __init__(self, base_path, splits, choices, model):
        self.utterances = self._getutterancelist(base_path, splits, choices, model)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        feature = torch.load(self.utterances[idx]).detach().cpu()
        feature = feature.squeeze()
        # print(len(speaker_feature))
        return feature, self.utterances[idx]
        
    def collate_fn(self, samples):
        return zip(*samples)

    def _getutterancelist(self, base_path, splits, choices, model):
        utterance_list = []

        split_pathes = [os.path.join(base_path, split) for split in splits]

        split_choices = [(math.ceil(choices*(i+1)/len(split_pathes))-math.ceil(choices*i/len(split_pathes))) for i in range(len(split_pathes))]
        for split_path, split_choice in zip(split_pathes, split_choices):
            split_utterance_list = []
            for speaker in glob.glob(os.path.join(split_path, "*[!.txt]")):
                speaker_features = []
                for chapter in glob.glob(os.path.join(speaker, "*")):
                    for feature_path in glob.glob(os.path.join(chapter, f"{model}-*")):
                        split_utterance_list.append(feature_path)
            utterance_list += random.sample(split_utterance_list, k=min(split_choice, len(split_utterance_list)))
            
        # print(len(speaker_list))
        return utterance_list
    
class SpeakerLevelDatasetByUtterance(Dataset):
    def __init__(self, base_path, splits, choices, model):
        self.utterances = self._getutterancelist(base_path, splits, choices, model)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        feature = torch.load(self.utterances[idx]).detach().cpu()
        feature = feature.squeeze()
        # print(len(speaker_feature))
        return feature, self.utterances[idx]
        
    def collate_fn(self, samples):
        return zip(*samples)

    def _getutterancelist(self, base_path, splits, choices, model):
        utterance_list = []
        
        split_pathes = [os.path.join(base_path, split) for split in splits]

        split_choices = [(math.ceil(choices*(i+1)/len(split_pathes))-math.ceil(choices*i/len(split_pathes))) for i in range(len(split_pathes))]
        for split_path, split_choice in zip(split_pathes, split_choices):
            all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
            analyze_speakers = random.sample(all_speakers, k=min(split_choice, len(all_speakers)))
            for speaker in analyze_speakers:
                for chapter in glob.glob(os.path.join(speaker, "*")):
                    for feature_path in glob.glob(os.path.join(chapter, f"{model}-*")):
                        utterance_list.append(feature_path)
            
        # print(len(speaker_list))
        return utterance_list

class ReconstructionBasedUtteranceLevelDataset(Dataset):
    def __init__(self, base_path, splits, choices, model, labels=None):
        self.model = model
        self.utterances = self._getutterancelist(base_path, splits, choices)
        self.base_path = base_path



    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        ssl_feature = torch.load(self.utterances[idx]).detach().cpu()
        ssl_feature = ssl_feature.squeeze()

        if self.model in ['hubert', 'wav2vec2']:
            mel_extractor = MelSpectrogram(n_fft=400, win_length=400, hop_length=320, center=False)
        elif self.model in ['modified_cpc']:
            mel_extractor = MelSpectrogram(n_fft=400, win_length=400, hop_length=160, center=False)
        else:
            mel_extractor = MelSpectrogram(n_fft=400, win_length=400, hop_length=160, center=True)

        wav_path = self.utterances[idx].replace(f"{self.model}-", "").replace(".pt", "").replace(self.base_path, "/groups/public/LibriSpeech/")
        wav, _ = apply_effects_file(
                wav_path,
                [
                    ["channels", "1"],
                    ["rate", "16000"],
                    ["norm"],
                ],
            )
        wav = wav.squeeze(0)

        mel_feature = mel_extractor(wav).reshape(-1, 128)
        length = min(len(mel_feature), len(ssl_feature))
        ssl_feature = ssl_feature[:length]
        mel_feature = mel_feature[:length]

        ssl_feature = torch.cat((ssl_feature[:length-4], ssl_feature[1:length-3], ssl_feature[2:length-2], ssl_feature[3:length-1], ssl_feature[4:length]), dim=1)
        mel_feature = mel_feature[2:length-2]
        # print(len(speaker_feature))
        return ssl_feature, mel_feature
        
    def collate_fn(self, samples):
        
        return zip(*samples)

    def _getutterancelist(self, base_path, splits, choices):
        utterance_list = []

        split_pathes = [os.path.join(base_path, split) for split in splits]

        split_choices = [(math.ceil(choices*(i+1)/len(split_pathes))-math.ceil(choices*i/len(split_pathes))) for i in range(len(split_pathes))]
        for split_path, split_choice in zip(split_pathes, split_choices):
            split_utterance_list = []
            for speaker in glob.glob(os.path.join(split_path, "*[!.txt]")):
                speaker_features = []
                for chapter in glob.glob(os.path.join(speaker, "*")):
                    for feature_path in glob.glob(os.path.join(chapter, f"{self.model}-*")):
                        split_utterance_list.append(feature_path)
                        
            utterance_list += random.sample(split_utterance_list, k=split_choice)
            
        # print(len(speaker_list))
        return utterance_list
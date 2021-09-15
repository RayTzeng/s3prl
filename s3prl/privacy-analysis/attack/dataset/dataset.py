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
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        return speaker_feature
        
    def collate_fn(self, samples):
        return samples

    def _getspeakerlist(self, base_path, splits, choices):
        speaker_list = []

        split_pathes = [os.path.join(base_path, split) for split in splits]

        split_choices = [(math.ceil(choices*(i+1)/len(split_pathes))-math.ceil(choices*i/len(split_pathes))) for i in range(len(split_pathes))]
        for split_path, split_choice in zip(split_pathes, split_choices):
            all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
            analyze_speakers = random.sample(all_speakers, k=split_choice)
            
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
        return feature
        
    def collate_fn(self, samples):
        return samples

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
            utterance_list += random.sample(split_utterance_list, k=split_choice)
            
        # print(len(speaker_list))
        return utterance_list
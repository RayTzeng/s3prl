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
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class SpeakerLevelDataset(Dataset):
    def __init__(self, base_path, seen_splits, unseen_splits, choices, model):
        self.data = self._getdatalist(
            base_path, seen_splits, unseen_splits, choices, model
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_x_path, feature_y_path, label, speaker = self.data[idx]
        feature_x = torch.load(feature_x_path).detach().cpu()
        feature_x = feature_x.squeeze()
        feature_y = torch.load(feature_y_path).detach().cpu()
        feature_y = feature_y.squeeze()

        return feature_x.numpy(), feature_y.numpy(), label, speaker

    def collate_fn(self, samples):
        return zip(*samples)

    def _getdatalist(self, base_path, seen_splits, unseen_splits, choices, model):
        data_list = []

        seen_split_pathes = [os.path.join(base_path, split) for split in seen_splits]
        unseen_split_pathes = [
            os.path.join(base_path, split) for split in unseen_splits
        ]

        split_choices = [
            (
                math.ceil(choices * (i + 1) / len(seen_split_pathes))
                - math.ceil(choices * i / len(seen_split_pathes))
            )
            for i in range(len(seen_split_pathes))
        ]
        for split_path, split_choice in (seen_split_pathes, split_choices):
            all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
            analyze_speakers = random.sample(all_speakers, k=split_choice)
            for speaker in analyze_speakers:
                for chapter in glob.glob(os.path.join(speaker, "*")):
                    feature_pathes = glob.glob(os.path.join(chapter, f"{model}-*"))
                    for i in range(len(feature_pathes) - 1):
                        data_list.append(
                            (
                                feature_pathes[i],
                                feature_pathes[i + 1],
                                1,
                                speaker.split("/")[-1],
                            )
                        )
        print(len(data_list))

        split_choices = [
            (
                math.ceil(choices * (i + 1) / len(unseen_split_pathes))
                - math.ceil(choices * i / len(unseen_split_pathes))
            )
            for i in range(len(unseen_split_pathes))
        ]
        for split_path, split_choice in (unseen_split_pathes, split_choices):
            all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
            analyze_speakers = random.sample(all_speakers, k=split_choice)
            for speaker in analyze_speakers:
                for chapter in glob.glob(os.path.join(speaker, "*")):
                    feature_pathes = glob.glob(os.path.join(chapter, f"{model}-*"))
                    for i in range(len(feature_pathes) - 1):
                        data_list.append(
                            (
                                feature_pathes[i],
                                feature_pathes[i + 1],
                                0,
                                speaker.split("/")[-1],
                            )
                        )
        print(len(data_list))

        return data_list

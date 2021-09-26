import glob
import math
import os
import random

import torch
from torch.utils.data.dataset import Dataset


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
        
        features_x, features_y, labels, speakers = [], [], [], []

        for feature_x, feature_y, label, speaker in samples:
            features_x.append(feature_x), features_y, labels, speakers
            features_y.append(feature_y)
            labels.append(label)
            speakers.append(speaker)

        return features_x, features_y, labels, speakers

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
        for split_path, split_choice in zip(seen_split_pathes, split_choices):
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
        for split_path, split_choice in zip(unseen_split_pathes, split_choices):
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

class UtteranceLevelDataset(Dataset):
    def __init__(self, base_path, seen_splits, unseen_splits, choices, model):
        self.utterances = self._getutterancelist(
            base_path, seen_splits, unseen_splits, choices, model
        )

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        feature_path, label = self.utterances[idx]
        feature = torch.load(feature_path).detach().cpu()
        feature = feature.squeeze()
        # print(len(speaker_feature))
        return feature, label
        
    def collate_fn(self, samples):
        return zip(*samples)

    def _getutterancelist(self, base_path, seen_splits, unseen_splits, choices, model):
        utterance_list = []

        seen_split_pathes = [os.path.join(base_path, split) for split in seen_splits]
        unseen_split_pathes = [
            os.path.join(base_path, split) for split in unseen_splits
        ]

        # seen
        split_choices = [
            (
                math.ceil(choices * (i + 1) / len(seen_split_pathes))
                - math.ceil(choices * i / len(seen_split_pathes))
            )
            for i in range(len(seen_split_pathes))
        ]
        for split_path, split_choice in zip(seen_split_pathes, split_choices):
            split_utterance_list = []
            for speaker in glob.glob(os.path.join(split_path, "*[!.txt]")):
                speaker_features = []
                for chapter in glob.glob(os.path.join(speaker, "*")):
                    for feature_path in glob.glob(os.path.join(chapter, f"{model}-*")):
                        split_utterance_list.append((feature_path, 0))
            utterance_list += random.sample(split_utterance_list, k=split_choice)

        # unseen
        split_choices = [
            (
                math.ceil(choices * (i + 1) / len(unseen_split_pathes))
                - math.ceil(choices * i / len(unseen_split_pathes))
            )
            for i in range(len(unseen_split_pathes))
        ]
        for split_path, split_choice in zip(unseen_split_pathes, split_choices):
            split_utterance_list = []
            for speaker in glob.glob(os.path.join(split_path, "*[!.txt]")):
                speaker_features = []
                for chapter in glob.glob(os.path.join(speaker, "*")):
                    for feature_path in glob.glob(os.path.join(chapter, f"{model}-*")):
                        split_utterance_list.append((feature_path, 1))
            utterance_list += random.sample(split_utterance_list, k=split_choice)
            
        # print(len(speaker_list))
        return utterance_list

class CertainSpeakerDataset(Dataset):
    def __init__(self, base_path, positive_speakers, negative_speakers, model):
        self.data = self._getdatalist(
            base_path, positive_speakers, negative_speakers, model
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

    def _getdatalist(self, base_path, positive_speakers, negative_speakers, model):
        data_list = []


        for speaker in positive_speakers:
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

        for speaker in negative_speakers:
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


class CertainUtteranceDataset(Dataset):
    def __init__(self, base_path, positive_utterances, negative_utterances, model):
        self.utterances = self._getutterancelist(
            base_path, positive_utterances, negative_utterances, model
        )

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        feature_path, label = self.utterances[idx]
        feature = torch.load(feature_path).detach().cpu()
        feature = feature.squeeze()
        # print(len(speaker_feature))
        return feature, label
        
    def collate_fn(self, samples):
        return zip(*samples)

    def _getutterancelist(self, base_path, positive_utterances, negative_utterances, model):
        utterance_list = []

        for utterance in positive_utterances:
            utterance_list.append((utterance, 0))

        for utterance in negative_utterances:
            utterance_list.append((utterance, 1))

            
        # print(len(speaker_list))
        return utterance_list
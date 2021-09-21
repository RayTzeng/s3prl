import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import scikitplot as skplt
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from dataset.dataset import SpeakerLevelDataset
from utils.utils import compute_speaker_adversarial_advantage_by_percentile, compute_speaker_adversarial_advantage_by_ROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)
    CHOICE_SIZE = args.speaker_choice_size

    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]

    seen_dataset = SpeakerLevelDataset(
        args.base_path, seen_splits, CHOICE_SIZE, args.model
    )
    unseen_dataset = SpeakerLevelDataset(
        args.base_path, unseen_splits, CHOICE_SIZE, args.model
    )

    seen_dataloader = DataLoader(
        seen_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=seen_dataset.collate_fn,
    )
    unseen_dataloader = DataLoader(
        unseen_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=unseen_dataset.collate_fn,
    )

    intra_seen_speaker_sim = defaultdict(float)

    # seen data
    for batch_id, (speaker_features, speakers) in enumerate(
        tqdm(seen_dataloader, dynamic_ncols=True, desc="Unseen")
    ):
        for i in range(len(speakers)):
            speaker_feature = speaker_features[i]
            sim = cosine_similarity(speaker_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            intra_seen_speaker_sim[speakers[i]] = np.mean(sim)

    seen_spkr_sim = []
    seen_spkr_list = []
    for k, v in intra_seen_speaker_sim.items():
        seen_spkr_sim.append(v)
        seen_spkr_list.append(k)

    df = pd.DataFrame(
        {
            "Seen_spkr": seen_spkr_list,
            "Similarity": seen_spkr_sim
        }
    )
    df.to_csv(
        os.path.join(args.output_path, f"{args.model}-seen-speaker-similarity.csv"),
        index=False,
    )


    intra_unseen_speaker_sim = defaultdict(list)

    # unseen data
    for batch_id, (speaker_features, speakers) in enumerate(
        tqdm(unseen_dataloader, dynamic_ncols=True, desc="Unseen")
    ):
        for i in range(len(speakers)):
            speaker_feature = speaker_features[i]
            sim = cosine_similarity(speaker_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            intra_unseen_speaker_sim[speakers[i]] = np.mean(sim)

    unseen_spkr_sim = []
    unseen_spkr_list = []
    for k, v in intra_unseen_speaker_sim.items():
        unseen_spkr_sim.append(v)
        unseen_spkr_list.append(k)

    df = pd.DataFrame(
        {
            "Unseen_spkr": unseen_spkr_list,
            "Similarity": unseen_spkr_sim
        }
    )
    df.to_csv(
        os.path.join(args.output_path, f"{args.model}-unseen-speaker-similarity.csv"),
        index=False,
    )


    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", help="directory of feature of LibriSpeech dataset"
    )
    parser.add_argument("--output_path", help="directory to save the analysis results")
    parser.add_argument(
        "--model", help="which self-supervised model you used to extract features"
    )
    parser.add_argument("--seed", type=int, default=57, help="random seed")
    parser.add_argument(
        "--speaker_choice_size", type=int, default=100, help="how many speaker to pick"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    args = parser.parse_args()

    main(args)

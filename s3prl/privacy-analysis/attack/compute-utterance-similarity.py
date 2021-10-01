import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from dataset.dataset import UtteranceLevelDataset
from utils.utils import compute_utterance_adversarial_advantage_by_percentile, compute_utterance_adversarial_advantage_by_ROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)
    CHOICE_SIZE = args.utterance_choice_size

    seen_splits = ["train-clean-100"]
    if args.unseen == 'LibriSpeech':
        unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]
    elif args.unseen == 'VCTK':
        unseen_splits = ['wav48']
    #unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]

    seen_dataset = UtteranceLevelDataset(
        args.base_path, seen_splits, CHOICE_SIZE, args.model
    )
    unseen_dataset = UtteranceLevelDataset(
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

    intra_seen_utterance_sim = defaultdict(float)

    # seen data
    for batch_id, (utterance_features, utterances) in enumerate(
        tqdm(seen_dataloader, dynamic_ncols=True, desc="Unseen")
    ):
        for i in range(len(utterances)):
            utterance_feature = utterance_features[i]
            sim = cosine_similarity(utterance_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            intra_seen_utterance_sim[utterances[i]] = np.mean(sim)

    seen_uttr_sim = []
    seen_uttr_list = []
    for k, v in intra_seen_utterance_sim.items():
        seen_uttr_sim.append(v)
        seen_uttr_list.append(k)

    df = pd.DataFrame(
        {
            "Seen_uttr": seen_uttr_list,
            "Similarity": seen_uttr_sim
        }
    )
    df.to_csv(
        os.path.join(args.output_path, f"{args.model}-seen-utterance-similarity.csv"),
        index=False,
    )



    intra_unseen_utterance_sim = defaultdict(float)
    # seen data
    for batch_id, (utterance_features, utterances) in enumerate(
        tqdm(unseen_dataloader, dynamic_ncols=True, desc="Unseen")
    ):
        for i in range(len(utterances)):
            utterance_feature = utterance_features[i]
            sim = cosine_similarity(utterance_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            intra_unseen_utterance_sim[utterances[i]] = np.mean(sim)

    unseen_uttr_sim = []
    unseen_uttr_list = []
    for k, v in intra_unseen_utterance_sim.items():
        unseen_uttr_sim.append(v)
        unseen_uttr_list.append(k)

    df = pd.DataFrame(
        {
            "Unseen_uttr": unseen_uttr_list,
            "Similarity": unseen_uttr_sim
        }
    )
    df.to_csv(
        os.path.join(args.output_path, f"{args.model}-unseen-utterance-similarity.csv"),
        index=False,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", help="directory of feature of LibriSpeech dataset"
    )
    parser.add_argument("--output_path", help="directory to save the analysis results")
    parser.add_argument(
        "--unseen", default="LibriSpeech", help="unseen data to use (LibriSpeech or VCTK)"
    )
    parser.add_argument(
        "--model", help="which self-supervised model you used to extract features"
    )
    parser.add_argument("--seed", type=int, default=57, help="random seed")
    parser.add_argument(
        "--utterance_choice_size",
        type=int,
        default=10000,
        help="how many speaker to pick",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    args = parser.parse_args()

    main(args)

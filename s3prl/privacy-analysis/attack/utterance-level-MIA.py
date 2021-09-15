import argparse
import glob
import math
import os
import random
import time

import IPython
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.dataset import UtteranceLevelDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)
    CHOICE_SIZE = args.utterance_choice_size

    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]
    
    seen_dataset = UtteranceLevelDataset(args.base_path, seen_splits, CHOICE_SIZE, args.model)
    unseen_dataset = UtteranceLevelDataset(args.base_path, unseen_splits, CHOICE_SIZE, args.model)

    seen_dataloader = DataLoader(seen_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=seen_dataset.collate_fn)
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=unseen_dataset.collate_fn)

    context_level_sim = []

    # seen data
    for batch_id, (utterance_features) in enumerate(tqdm(seen_dataloader, dynamic_ncols=True, desc=f'Seen')):
        for utterance_feature in utterance_features:
            sim = cosine_similarity(utterance_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            context_level_sim.append(np.mean(sim))


    # unseen data
    for batch_id, (utterance_features) in enumerate(tqdm(unseen_dataloader, dynamic_ncols=True, desc=f'Unseen')):
        for utterance_feature in utterance_features:
            sim = cosine_similarity(utterance_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            context_level_sim.append(np.mean(sim))


    # apply attack
    percentile_choice = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    seen_uttr_sim = context_level_sim[:CHOICE_SIZE]
    unseen_uttr_sim = context_level_sim[CHOICE_SIZE:]
    recall_by_percentile = []
    precision_by_percentile = []
    accuracy_by_percentile = []

    for percentile in percentile_choice:
        sorted_unseen_uttr_sim = sorted(unseen_uttr_sim)
        threshold = sorted_unseen_uttr_sim[
            math.floor(CHOICE_SIZE * percentile / 100)
        ]
        TP = len([sim for sim in seen_uttr_sim if sim < threshold])
        FN = len([sim for sim in seen_uttr_sim if sim >= threshold])
        FP = len([sim for sim in unseen_uttr_sim if sim < threshold])
        TN = len([sim for sim in unseen_uttr_sim if sim >= threshold])

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / (TP + FP + FN + TN)

        recall_by_percentile.append(recall)
        precision_by_percentile.append(precision)
        accuracy_by_percentile.append(accuracy)

    print()
    print(f"[{args.model}]")
    print(f"precentile: ", " | ".join(f"{num:5}%" for num in percentile_choice))
    print("-----------------------------------------------------------------")
    print(f"recall:     ", " | ".join(f"{num:.4f}" for num in recall_by_percentile))
    print(f"precision:  ", " | ".join(f"{num:.4f}" for num in precision_by_percentile))
    print(f"accuracy:   ", " | ".join(f"{num:.4f}" for num in accuracy_by_percentile))
    print()

    df = pd.DataFrame({'percentile': percentile_choice,
                        'recall': recall_by_percentile,
                        'precision': precision_by_percentile,
                        'accuracy': accuracy_by_percentile})
    df.to_csv(os.path.join(args.output_path, f"{args.model}-utterance-level-attack-result.csv"), index=False)


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
        "--utterance_choice_size", type=int, default=2000, help="how many speaker to pick"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers"
    )
    args = parser.parse_args()

    main(args)

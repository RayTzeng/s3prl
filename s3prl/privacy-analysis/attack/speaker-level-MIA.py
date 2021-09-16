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

from dataset.dataset import SpeakerLevelDataset

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

    intra_speaker_sim_mean = []
    colors = []

    # seen data
    for batch_id, (speaker_features) in enumerate(
        tqdm(seen_dataloader, dynamic_ncols=True, desc="Seen")
    ):
        for speaker_feature in speaker_features:
            # IPython.embed()
            sim = cosine_similarity(speaker_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            intra_speaker_sim_mean.append(np.mean(sim))
            colors.append("blue")

    # unseen data
    for batch_id, (speaker_features) in enumerate(
        tqdm(unseen_dataloader, dynamic_ncols=True, desc="Unseen")
    ):
        for speaker_feature in speaker_features:
            sim = cosine_similarity(speaker_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            intra_speaker_sim_mean.append(np.mean(sim))
            colors.append("red")

    plt.figure(figsize=(80, 40))
    plt.rcParams.update({"font.size": 40})

    low = min(intra_speaker_sim_mean)
    high = max(intra_speaker_sim_mean)
    plt.ylim([low - 0.25 * (high - low), high + 0.25 * (high - low)])

    x = [
        1,
        CHOICE_SIZE + 1,
    ]
    ticks = ["seen", "unseen"]
    plt.bar(
        range(1, len(intra_speaker_sim_mean) + 1), intra_speaker_sim_mean, color=colors
    )
    plt.xticks(x, ticks)
    plt.ylabel("Average similarity")
    plt.title("speaker similarity of {}".format(args.model))

    plt.plot(
        [0, len(intra_speaker_sim_mean) + 1],
        [np.mean(intra_speaker_sim_mean[:CHOICE_SIZE]) for _ in range(2)],
        ls="--",
        color="blue",
    )
    plt.plot(
        [0, len(intra_speaker_sim_mean) + 1],
        [np.mean(intra_speaker_sim_mean[CHOICE_SIZE:]) for _ in range(2)],
        ls="--",
        color="red",
    )
    plt.savefig(
        os.path.join(args.output_path, f"{args.model}-speaker-level-sim-bar-plot.png")
    )

    # apply attack
    percentile_choice = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    seen_spkr_sim = intra_speaker_sim_mean[:CHOICE_SIZE]
    unseen_spkr_sim = intra_speaker_sim_mean[CHOICE_SIZE:]
    recall_by_percentile = []
    precision_by_percentile = []
    accuracy_by_percentile = []

    for percentile in percentile_choice:
        sorted_unseen_spkr_sim = sorted(unseen_spkr_sim)
        threshold = sorted_unseen_spkr_sim[math.floor(CHOICE_SIZE * percentile / 100)]
        TP = len([sim for sim in seen_spkr_sim if sim > threshold])
        FN = len([sim for sim in seen_spkr_sim if sim <= threshold])
        FP = len([sim for sim in unseen_spkr_sim if sim > threshold])
        TN = len([sim for sim in unseen_spkr_sim if sim <= threshold])

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / (TP + FP + FN + TN)

        recall_by_percentile.append(recall)
        precision_by_percentile.append(precision)
        accuracy_by_percentile.append(accuracy)

    print()
    print(f"[{args.model}]")
    print("precentile: ", " | ".join(f"{num:5}%" for num in percentile_choice))
    print("-----------------------------------------------------------------")
    print("recall:     ", " | ".join(f"{num:.4f}" for num in recall_by_percentile))
    print("precision:  ", " | ".join(f"{num:.4f}" for num in precision_by_percentile))
    print("accuracy:   ", " | ".join(f"{num:.4f}" for num in accuracy_by_percentile))
    print()

    df = pd.DataFrame(
        {
            "percentile": percentile_choice,
            "recall": recall_by_percentile,
            "precision": precision_by_percentile,
            "accuracy": accuracy_by_percentile,
        }
    )
    df.to_csv(
        os.path.join(args.output_path, f"{args.model}-speaker-level-attack-result.csv"),
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

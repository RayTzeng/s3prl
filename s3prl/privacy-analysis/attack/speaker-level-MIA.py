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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)
    CHOICE_SIZE = args.speaker_choice_size

    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]
    seen_plot_color = ["blue"]
    unseen_plot_color = ["purple", "red", "orange", "yellow"]
    seen_split_pathes = [os.path.join(args.base_path, split) for split in seen_splits]
    unseen_split_pathes = [
        os.path.join(args.base_path, split) for split in unseen_splits
    ]

    intra_speaker_sim_mean = []
    colors = []

    # seen data
    for split_path, plot_color in zip(seen_split_pathes, seen_plot_color):
        all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
        analyze_speakers = random.sample(all_speakers, k=CHOICE_SIZE)
        for speaker in tqdm(analyze_speakers):

            # calculate intra speaker similarity

            speaker_features = []
            for chapter in glob.glob(os.path.join(speaker, "*")):
                for feature_path in glob.glob(os.path.join(chapter, f"{args.model}-*")):
                    feature = torch.load(feature_path).detach().cpu()
                    feature = feature.squeeze()
                    speaker_features.append(np.array(feature).mean(axis=0))

            sim = cosine_similarity(speaker_features)
            sim = sim[np.triu_indices(len(sim), k=1)]
            intra_speaker_sim_mean.append(np.mean(sim))
            colors.append(plot_color)

    # unseen data
    N = len(unseen_splits)
    for split_path, plot_color in zip(unseen_split_pathes, unseen_plot_color):
        all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
        analyze_speakers = random.sample(all_speakers, k=int(CHOICE_SIZE / N))
        for speaker in tqdm(analyze_speakers):

            # calculate intra speaker similarity

            speaker_features = []
            for chapter in glob.glob(os.path.join(speaker, "*")):
                for feature_path in glob.glob(os.path.join(chapter, f"{args.model}-*")):
                    feature = torch.load(feature_path).detach().cpu()
                    feature = feature.squeeze()
                    speaker_features.append(np.array(feature).mean(axis=0))

            sim = cosine_similarity(speaker_features)
            sim = sim[np.triu_indices(len(sim), k=1)]
            intra_speaker_sim_mean.append(np.mean(sim))
            colors.append(plot_color)

    plt.figure(figsize=(80, 40))
    plt.rcParams.update({"font.size": 40})

    low = min(intra_speaker_sim_mean)
    high = max(intra_speaker_sim_mean)
    plt.ylim([low - 0.25 * (high - low), high + 0.25 * (high - low)])

    x = [
        1,
        CHOICE_SIZE + 1,
        CHOICE_SIZE * ((N + 1) / N) + 1,
        CHOICE_SIZE * ((N + 2) / N) + 1,
        CHOICE_SIZE * ((N + 3) / N) + 1,
    ]
    ticks = np.concatenate((seen_splits, unseen_splits))
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
    df.to_csv(os.path.join(args.output_path, f"{args.model}-speaker-level-attack-result.csv"), index=False)
        


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
        "--speaker_choice_size", type=int, default=120, help="how many speaker to pick"
    )
    args = parser.parse_args()

    main(args)

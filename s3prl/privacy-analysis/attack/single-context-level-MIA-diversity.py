import argparse
import glob
import math
import os
import random
import time

import IPython
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPEAKER_CHOICE_SIZE = 20
random.seed(57)


def main(args):
    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]
    seen_plot_color = ["blue"]
    unseen_plot_color = ["purple", "red", "orange", "yellow"]
    seen_split_pathes = [os.path.join(args.base_path, split) for split in seen_splits]
    unseen_split_pathes = [
        os.path.join(args.base_path, split) for split in unseen_splits
    ]

    context_level_diversity = []
    prefix = [0]
    colors = []

    # seen data
    for split_path, plot_color in zip(seen_split_pathes, seen_plot_color):
        all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
        analyze_speakers = random.choices(all_speakers, k=SPEAKER_CHOICE_SIZE)
        for speaker in tqdm(analyze_speakers):

            # calculate pairwise-context-level similarity

            for chapter in glob.glob(os.path.join(speaker, "*")):
                for feature_path in glob.glob(os.path.join(chapter, f"{args.model}-*")):
                    feature = np.array(
                        torch.load(feature_path).detach().cpu().squeeze()
                    )
                    sim = cosine_similarity(feature)
                    sim = sim[np.triu_indices(len(sim), k=1)]
                    div = sim.std()
                    context_level_diversity.append(div)

            prefix.append(len(context_level_diversity))
            colors.append(plot_color)

    num_seen_uttr = prefix[-1]

    # unseen data
    N = len(unseen_splits)
    for split_path, plot_color in zip(unseen_split_pathes, unseen_plot_color):
        all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
        analyze_speakers = random.choices(all_speakers, k=int(SPEAKER_CHOICE_SIZE / N))
        for speaker in tqdm(analyze_speakers):

            # calculate pairwise-context-level similarity

            for chapter in glob.glob(os.path.join(speaker, "*")):
                for feature_path in glob.glob(os.path.join(chapter, f"{args.model}-*")):
                    feature = np.array(
                        torch.load(feature_path).detach().cpu().squeeze()
                    )
                    sim = cosine_similarity(feature)
                    sim = sim[np.triu_indices(len(sim), k=1)]
                    div = sim.std()
                    context_level_diversity.append(div)

            prefix.append(len(context_level_diversity))
            colors.append(plot_color)

    num_unseen_uttr = prefix[-1] - num_seen_uttr

    mean_context_level_div = [
        np.mean(context_level_diversity[prefix[i] : prefix[i + 1]])
        for i in range(len(prefix) - 1)
    ]
    max_context_level_div = [
        max(context_level_diversity[prefix[i] : prefix[i + 1]])
        - mean_context_level_div[i]
        for i in range(len(prefix) - 1)
    ]
    min_context_level_div = [
        mean_context_level_div[i]
        - min(context_level_diversity[prefix[i] : prefix[i + 1]])
        for i in range(len(prefix) - 1)
    ]

    plt.figure(figsize=(80, 40))
    plt.rcParams.update({"font.size": 40})

    low = min(context_level_diversity)
    high = max(context_level_diversity)
    plt.ylim([low - 0.25 * (high - low), high + 0.25 * (high - low)])

    x = [
        1,
        SPEAKER_CHOICE_SIZE + 1,
        SPEAKER_CHOICE_SIZE * ((N + 1) / N) + 1,
        SPEAKER_CHOICE_SIZE * ((N + 2) / N) + 1,
        SPEAKER_CHOICE_SIZE * ((N + 3) / N) + 1,
    ]
    ticks = np.concatenate((seen_splits, unseen_splits))
    plt.bar(
        range(1, len(mean_context_level_div) + 1),
        mean_context_level_div,
        yerr=[min_context_level_div, max_context_level_div],
        color=colors,
    )
    plt.xticks(x, ticks)
    plt.ylabel("Average Standard Deviation")
    plt.title("Context-level diversity of {}".format(args.model))

    plt.plot(
        [0, len(mean_context_level_div) + 1],
        [np.mean(context_level_diversity[:num_seen_uttr]) for _ in range(2)],
        ls="--",
        color="blue",
    )
    plt.plot(
        [0, len(mean_context_level_div) + 1],
        [np.mean(context_level_diversity[num_seen_uttr:]) for _ in range(2)],
        ls="--",
        color="red",
    )
    plt.savefig(
        os.path.join(
            args.output_path, f"{args.model}-single-context-diversity-bar-plot.png"
        )
    )

    # apply attack
    percentile_choice = [50, 60, 70, 80, 90]

    seen_spkr_div = context_level_diversity[:num_seen_uttr]
    unseen_spkr_div = context_level_diversity[num_seen_uttr:]
    recall_by_percentile = []
    precision_by_percentile = []
    accuracy_by_percentile = []

    for percentile in percentile_choice:
        sorted_unseen_spkr_div = sorted(unseen_spkr_div)
        threshold = sorted_unseen_spkr_div[
            math.floor(num_unseen_uttr * percentile / 100)
        ]
        TP = len([div for div in seen_spkr_div if div > threshold])
        FN = len([div for div in seen_spkr_div if div <= threshold])
        FP = len([div for div in unseen_spkr_div if div > threshold])
        TN = len([div for div in unseen_spkr_div if div <= threshold])

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", help="directory of feature of LibriSpeech dataset"
    )
    parser.add_argument("--output_path", help="directory to save the analysis results")
    parser.add_argument(
        "--model", help="which self-supervised model you used to extract features"
    )
    args = parser.parse_args()

    main(args)
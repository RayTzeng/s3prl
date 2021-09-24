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

from dataset.dataset import UtteranceLevelDataset
from utils.utils import compute_utterance_adversarial_advantage_by_percentile, compute_utterance_adversarial_advantage_by_ROC
from utils.utils import compute_speaker_adversarial_advantage_by_percentile, compute_speaker_adversarial_advantage_by_ROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)
    CHOICE_SIZE = args.utterance_choice_size

    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]

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

    context_level_sim = []

    # seen data
    for batch_id, (utterance_features, utterances) in enumerate(
        tqdm(seen_dataloader, dynamic_ncols=True, desc="Seen")
    ):
        for utterance_feature in utterance_features:
            sim = cosine_similarity(utterance_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            context_level_sim.append(np.mean(sim))

    # unseen data
    for batch_id, (utterance_features, utterances) in enumerate(
        tqdm(unseen_dataloader, dynamic_ncols=True, desc="Unseen")
    ):
        for utterance_feature in utterance_features:
            sim = cosine_similarity(utterance_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            context_level_sim.append(np.mean(sim))

    # apply attack
    percentile_choice = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    seen_uttr_sim = context_level_sim[:CHOICE_SIZE]
    unseen_uttr_sim = context_level_sim[CHOICE_SIZE:]

    if np.mean(unseen_uttr_sim) <= 0.5:
        TPR, FPR, AA = compute_utterance_adversarial_advantage_by_percentile(seen_uttr_sim, unseen_uttr_sim, percentile_choice, args.model)
    else:
        TPR, FPR, AA = compute_speaker_adversarial_advantage_by_percentile(seen_uttr_sim, unseen_uttr_sim, percentile_choice, args.model)

    df = pd.DataFrame(
        {
            "Percentile": percentile_choice,
            "True Positive Rate": TPR,
            "False Positive Rate": FPR,
            "Adversarial Advantage": AA,
        }
    )
    df.to_csv(
        os.path.join(args.output_path, f"{args.model}-utterance-level-attack-result-by-percentile.csv"),
        index=False,
    )

    if np.mean(unseen_uttr_sim) <= 0.5:
        TPRs, FPRs, avg_AUC = compute_utterance_adversarial_advantage_by_ROC(seen_uttr_sim, unseen_uttr_sim, args.model)
    else:
        TPRs, FPRs, avg_AUC = compute_speaker_adversarial_advantage_by_ROC(seen_uttr_sim, unseen_uttr_sim, args.model)

    plt.figure()
    plt.rcParams.update({"font.size": 12})
    plt.title(f'Utterance-level attack ROC Curve - {args.model}')
    plt.plot(FPRs, TPRs, color="darkorange", lw=2, label=f"ROC curve (area = {avg_AUC:0.2f})")
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(args.output_path, f"{args.model}-utterance-level-attack-ROC-curve.png")
    )

    df = pd.DataFrame(
        {
            "Seen_uttr_sim": seen_uttr_sim, 
            "Unseen_uttr_sim": unseen_uttr_sim
        }
    )
    df.to_csv(
        os.path.join(args.output_path, f"{args.model}-utterance-level-attack-similarity.csv"),
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
        "--utterance_choice_size",
        type=int,
        default=10000,
        help="how many speaker to pick",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    args = parser.parse_args()

    main(args)

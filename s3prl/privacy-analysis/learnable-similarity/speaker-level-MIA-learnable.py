import argparse
import math
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import SpeakerLevelDataset
from model.learnable_similarity_model import SpeakerLevelModel
from utils.utils import compute_speaker_adversarial_advantage_by_percentile, compute_speaker_adversarial_advantage_by_ROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)
    CHOICE_SIZE = args.speaker_choice_size

    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]

    seen_dataset = SpeakerLevelDataset(
        args.base_path, seen_splits, [], CHOICE_SIZE, args.model
    )
    unseen_dataset = SpeakerLevelDataset(
        args.base_path, [], unseen_splits, CHOICE_SIZE, args.model
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

    ckpt = torch.load(args.sim_model_path)
    sim_predictor = SpeakerLevelModel(ckpt["linear.weight"].shape[0]).to(device)
    sim_predictor.load_state_dict(ckpt)
    sim_predictor.eval()

    # seen data

    seen_speaker_sim = defaultdict(list)

    with torch.no_grad():
        for batch_id, (features_x, features_y, labels, speakers) in enumerate(
            tqdm(seen_dataloader, dynamic_ncols=True, desc="Seen")
        ):
            features_x = [torch.FloatTensor(feature).to(device) for feature in features_x]
            features_y = [torch.FloatTensor(feature).to(device) for feature in features_y]
            labels = torch.FloatTensor([label for label in labels]).to(device)
            
            pred = sim_predictor(features_x, features_y)
            for i in range(len(speakers)):
                seen_speaker_sim[speakers[i]].append(pred[i].cpu().item())

    unseen_speaker_sim = defaultdict(list)

    with torch.no_grad():
        for batch_id, (features_x, features_y, labels, speakers) in enumerate(
            tqdm(unseen_dataloader, dynamic_ncols=True, desc="Unseen")
        ):
            features_x = [torch.FloatTensor(feature).to(device) for feature in features_x]
            features_y = [torch.FloatTensor(feature).to(device) for feature in features_y]
            labels = torch.FloatTensor([label for label in labels]).to(device)
            
            pred = sim_predictor(features_x, features_y)
            for i in range(len(speakers)):
                unseen_speaker_sim[speakers[i]].append(pred[i].cpu().item())

    intra_speaker_sim_mean = []
    colors = []

    for k, v in seen_speaker_sim.items():
        intra_speaker_sim_mean.append(np.mean(v))
        colors.append("blue")

    for k, v in unseen_speaker_sim.items():
        intra_speaker_sim_mean.append(np.mean(v))
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
        os.path.join(
            args.output_path, f"{args.model}-speaker-level-learnable-sim-bar-plot.png"
        )
    )

    # apply attack
    percentile_choice = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    seen_spkr_sim = intra_speaker_sim_mean[:CHOICE_SIZE]
    unseen_spkr_sim = intra_speaker_sim_mean[CHOICE_SIZE:]

    TPR, FPR, AA = compute_speaker_adversarial_advantage_by_percentile(seen_spkr_sim, unseen_spkr_sim, percentile_choice, args.model)

    df = pd.DataFrame(
        {
            "Percentile": percentile_choice,
            "True Positive Rate": TPR,
            "False Positive Rate": FPR,
            "Adversarial Advantage": AA,
        }
    )
    df.to_csv(
        os.path.join(args.output_path, f"{args.model}-speaker-level-attack-result-by-percentile.csv"),
        index=False,
    )

    TPRs, FPRs, avg_AUC = compute_speaker_adversarial_advantage_by_ROC(seen_spkr_sim, unseen_spkr_sim, args.model)

    plt.figure()
    plt.rcParams.update({"font.size": 12})
    plt.title(f'Speaker-level attack ROC Curve - {args.model}')
    plt.plot(FPRs, TPRs, color="darkorange", lw=2, label=f"ROC curve (area = {avg_AUC:0.2f})")
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(args.output_path, f"{args.model}-speaker-level-attack-ROC-curve.png")
    )

    df = pd.DataFrame(
        {
            "Seen_spkr_sim": seen_spkr_sim,
            "Unseen_spkr_sim": unseen_spkr_sim
        }
    )
    df.to_csv(
        os.path.join(args.output_path, f"{args.model}-speaker-level-attack-similarity.csv"),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", help="directory of feature of LibriSpeech dataset"
    )
    parser.add_argument("--output_path", help="directory to save the analysis results")
    parser.add_argument("--sim_model_path", help="path of similarity model")
    parser.add_argument(
        "--model", help="which self-supervised model you used to extract features"
    )
    parser.add_argument("--seed", type=int, default=57, help="random seed")
    parser.add_argument(
        "--speaker_choice_size", type=int, default=100, help="how many speaker to pick"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers")

    args = parser.parse_args()

    main(args)

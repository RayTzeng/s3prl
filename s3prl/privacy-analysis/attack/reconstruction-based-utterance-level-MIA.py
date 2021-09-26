import warnings
import argparse
import math
import os
import random
import s3prl.hub as hub
import itertools


import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from audtorch.metrics.functional import pearsonr
import IPython

from dataset.dataset import ReconstructionBasedUtteranceLevelDataset
from utils.utils import compute_speaker_adversarial_advantage_by_percentile, compute_speaker_adversarial_advantage_by_ROC
from utils.CCAmodels import DCCAModel, DCCAEModel

from cca_zoo.deepmodels import objectives, architectures, DCCA,DCCAE

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)
    CHOICE_SIZE = args.utterance_choice_size // 2

    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]

    seen_dataset = ReconstructionBasedUtteranceLevelDataset(
        args.base_path, seen_splits, 100, args.model
    )
    all_unseen_dataset = ReconstructionBasedUtteranceLevelDataset(
        args.base_path, unseen_splits, CHOICE_SIZE + 100, args.model
    )

    unseen_dataset, cca_dataset = torch.utils.data.random_split(
        all_unseen_dataset, [100, CHOICE_SIZE])

    cca_dataloader = DataLoader(
        cca_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        collate_fn=all_unseen_dataset.collate_fn,
    )

    seen_dataloader = DataLoader(
        seen_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=seen_dataset.collate_fn,
    )

    unseen_dataloader = DataLoader(
        unseen_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=all_unseen_dataset.collate_fn,
    )

    latent_dims = 70
    x_dim = seen_dataset[0][0].shape[1]
    y_dim = seen_dataset[0][1].shape[1]

    model = DCCAEModel(x_dim=x_dim, y_dim=y_dim,
                     latent_dims=latent_dims).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(tqdm(cca_dataloader, dynamic_ncols=True, desc=f"Train | Epoch {epoch}")):
            # def closure():
            optimizer.zero_grad()
            x = torch.cat(x).to(device)
            y = torch.cat(y).to(device)
            loss = model.loss(x, y)
            loss.backward()
            tqdm.write(f"loss: {loss.item():.4f}")

            torch.nn.utils.clip_grad_value_(
                model.parameters(), clip_value=float("inf"))
            optimizer.step()
            # loss = closure()
            train_loss += loss.item() / len(cca_dataloader)

        epoch_train_loss = train_loss
        print('====> Epoch: {} Average train loss: {:.4f}'.format(
            epoch, epoch_train_loss))

        if epoch % 1 == 0:
            context_level_sim = []
            context_level_sim_2 = []

            model.eval()
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(tqdm(seen_dataloader, dynamic_ncols=True, desc=f"Seen")):
                    # length = [len(rep) for rep in x]
                    # prefix = list(itertools.accumulate(length, initial=0))
                    # x = torch.cat(x).to(device)
                    # y = torch.cat(y).to(device)
                    # z_x, z_y = model(x, y)

                    x = torch.cat(x).to(device)
                    y = torch.cat(y).to(device)
                    score = model.score(x, y)
                    context_level_sim.append(np.mean(score))

                    # for i in range(len(prefix) - 1):
                    #     corr = pearsonr(z_x[prefix[i]:prefix[i+1]], z_y[prefix[i]:prefix[i+1]])
                    #     context_level_sim.append(torch.mean(corr).cpu().item())

                    #     corr = pearsonr(z_x[prefix[i]:prefix[i+1]].T, z_y[prefix[i]:prefix[i+1]].T)
                    #     context_level_sim_2.append(torch.mean(corr).cpu().item())


                for batch_idx, (x, y) in enumerate(tqdm(unseen_dataloader, dynamic_ncols=True, desc=f"Unseen")):
                    # length = [len(rep) for rep in x]
                    # prefix = list(itertools.accumulate(length, initial=0))
                    # x = torch.cat(x).to(device)
                    # y = torch.cat(y).to(device)
                    # z_x, z_y = model(x, y)

                    x = torch.cat(x).to(device)
                    y = torch.cat(y).to(device)
                    score = model.score(x, y)
                    context_level_sim.append(np.mean(score))

                    # for i in range(len(prefix) - 1):
                    #     corr = pearsonr(z_x[prefix[i]:prefix[i+1]], z_y[prefix[i]:prefix[i+1]])
                    #     context_level_sim.append(torch.mean(corr).cpu().item())

                    #     corr = pearsonr(z_x[prefix[i]:prefix[i+1]].T, z_y[prefix[i]:prefix[i+1]].T)
                    #     context_level_sim_2.append(torch.mean(corr).cpu().item())

            percentile_choice = [10, 20, 30, 40, 50, 60, 70, 80, 90]

            print("feature correlation")
            
            seen_uttr_sim = context_level_sim[:100]
            unseen_uttr_sim = context_level_sim[100:]

            TPR, FPR, AA = compute_speaker_adversarial_advantage_by_percentile(
                seen_uttr_sim, unseen_uttr_sim, percentile_choice, args.model)
            TPRs, FPRs, avg_AUC = compute_speaker_adversarial_advantage_by_ROC(
                seen_uttr_sim, unseen_uttr_sim, args.model)

            # print("dimensional correlation")

            # seen_uttr_sim = context_level_sim_2[:100]
            # unseen_uttr_sim = context_level_sim_2[100:]

            # TPR, FPR, AA = compute_speaker_adversarial_advantage_by_percentile(
            #     seen_uttr_sim, unseen_uttr_sim, percentile_choice, args.model)
            # TPRs, FPRs, avg_AUC = compute_speaker_adversarial_advantage_by_ROC(
            #     seen_uttr_sim, unseen_uttr_sim, args.model)

        # model.eval()
        # for dataset in [seen_dataset, unseen_dataset]:
        #     for i in range(5):
        #         eval_dataloader = DataLoader(
        #             Subset(dataset, [i]),
        #             batch_size=1,
        #             shuffle=False,
        #             num_workers=1,
        #             collate_fn=dataset.collate_fn,
        #         )

    # apply attack
    percentile_choice = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    seen_uttr_sim = context_level_sim[:1000]
    unseen_uttr_sim = context_level_sim[1000:]

    TPR, FPR, AA = compute_speaker_adversarial_advantage_by_percentile(
        seen_uttr_sim, unseen_uttr_sim, percentile_choice, args.model)

    # df = pd.DataFrame(
    #     {
    #         "Percentile": percentile_choice,
    #         "True Positive Rate": TPR,
    #         "False Positive Rate": FPR,
    #         "Adversarial Advantage": AA,
    #     }
    # )
    # df.to_csv(
    #     os.path.join(args.output_path, f"{args.model}-recon-utterance-level-attack-result-by-percentile.csv"),
    #     index=False,
    # )

    TPRs, FPRs, avg_AUC = compute_speaker_adversarial_advantage_by_ROC(
        seen_uttr_sim, unseen_uttr_sim, args.model)

    # plt.figure()
    # plt.rcParams.update({"font.size": 12})
    # plt.title(f'Utterance-level attack ROC Curve - {args.model}')
    # plt.plot(FPRs, TPRs, color="darkorange", lw=2, label=f"ROC curve (area = {avg_AUC:0.2f})")
    # plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.legend(loc="lower right")
    # plt.savefig(
    #     os.path.join(args.output_path, f"{args.model}-recon-utterance-level-attack-ROC-curve.png")
    # )

    # df = pd.DataFrame(
    #     {
    #         "Seen_uttr_sim": seen_uttr_sim,
    #         "Unseen_uttr_sim": unseen_uttr_sim
    #     }
    # )
    # df.to_csv(
    #     os.path.join(args.output_path, f"{args.model}-recon-utterance-level-attack-similarity.csv"),
    #     index=False,
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", help="directory of feature of LibriSpeech dataset"
    )
    parser.add_argument(
        "--output_path", help="directory to save the analysis results")
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
    parser.add_argument("--batch_size", type=int,
                        default=64, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="number of workers")
    args = parser.parse_args()

    main(args)

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
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import IPython

from dataset.dataset import ReconstructionBasedUtteranceLevelDataset
from utils.utils import compute_utterance_adversarial_advantage_by_percentile, compute_utterance_adversarial_advantage_by_ROC
from utils.deepwrapper import DeepWrapper

from cca_zoo.deepmodels import objectives, architectures, DCCAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)
    CHOICE_SIZE = args.utterance_choice_size

    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]

    seen_dataset = ReconstructionBasedUtteranceLevelDataset(
        args.base_path, seen_splits, CHOICE_SIZE, args.model
    )
    unseen_dataset = ReconstructionBasedUtteranceLevelDataset(
        args.base_path, unseen_splits, CHOICE_SIZE, args.model
    )

    train_dataloader = DataLoader(
        unseen_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=8,
        collate_fn=unseen_dataset.collate_fn,
    )

    

    latent_dims = 32

    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=768)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=201)
    decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=768)
    decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=201)
    dccae_model = DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2]).to(device)

    optimizer = torch.optim.Adam(dccae_model.parameters(), lr=1e-4)
    epochs = 20

    for epoch in range(1, epochs + 1):
        dccae_model.train()
        train_loss = 0
        for batch_idx, (data, label) in enumerate(tqdm(train_dataloader, dynamic_ncols=True, desc=f"Train | Epoch {epoch}")):
            optimizer.zero_grad()
            data = [d.to(device) for d in list(data)]
            loss = dccae_model.loss(*data)
            loss.backward()
            torch.nn.utils.clip_grad_value_(dccae_model.parameters(), clip_value=float('inf'))
            optimizer.step()
            train_loss += loss.item()
        epoch_train_loss = train_loss / len(train_dataloader)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(
            epoch, epoch_train_loss))

        for dataset in [seen_dataset, unseen_dataset]:
            for i in range(5):
                eval_dataloader = DataLoader(
                    Subset(dataset, [i]),
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    collate_fn=dataset.collate_fn,
                )

                with torch.no_grad():
                    for batch_idx, (data, label) in enumerate(eval_dataloader):
                        data = [d.to(device) for d in list(data)]
                        z = dccae_model(*data)
                        if batch_idx == 0:
                            z_list = [z_i.detach().cpu().numpy() for i, z_i in enumerate(z)]
                        else:
                            z_list = [np.append(z_list[i], z_i.detach().cpu().numpy(), axis=0) for
                                    i, z_i in enumerate(z)]
                z_list = dccae_model.post_transform(*z_list, train=False)

                transformed_views = z_list
                all_corrs = []
                for x, y in itertools.product(transformed_views, repeat=2):
                    all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[:x.shape[1], y.shape[1]:]))
                all_corrs = np.array(all_corrs).reshape(
                    (len(transformed_views), len(transformed_views), -1))

                pair_corrs = all_corrs
                n_views = pair_corrs.shape[0]
                dim_corrs = (pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - n_views) / (
                        n_views ** 2 - n_views)


                print(np.mean(dim_corrs))


#     context_level_sim = []

#     # seen data
#     for batch_id, (utterance_features) in enumerate(
#         tqdm(seen_dataloader, dynamic_ncols=True, desc="Seen")
#     ):
#         for utterance_feature in utterance_features:
#             sim = cosine_similarity(utterance_feature)
#             sim = sim[np.triu_indices(len(sim), k=1)]
#             context_level_sim.append(np.mean(sim))

#     # unseen data
#     for batch_id, (utterance_features) in enumerate(
#         tqdm(unseen_dataloader, dynamic_ncols=True, desc="Unseen")
#     ):
#         for utterance_feature in utterance_features:
#             sim = cosine_similarity(utterance_feature)
#             sim = sim[np.triu_indices(len(sim), k=1)]
#             context_level_sim.append(np.mean(sim))

#     # apply attack
#     percentile_choice = [10, 20, 30, 40, 50, 60, 70, 80, 90]

#     seen_uttr_sim = context_level_sim[:CHOICE_SIZE]
#     unseen_uttr_sim = context_level_sim[CHOICE_SIZE:]

#     TPR, FPR, AA = compute_utterance_adversarial_advantage_by_percentile(seen_uttr_sim, unseen_uttr_sim, percentile_choice, args.model)

#     df = pd.DataFrame(
#         {
#             "Percentile": percentile_choice,
#             "True Positive Rate": TPR,
#             "False Positive Rate": FPR,
#             "Adversarial Advantage": AA,
#         }
#     )
#     df.to_csv(
#         os.path.join(args.output_path, f"{args.model}-utterance-level-attack-result-by-percentile.csv"),
#         index=False,
#     )

#     TPRs, FPRs, avg_AUC = compute_utterance_adversarial_advantage_by_ROC(seen_uttr_sim, unseen_uttr_sim, args.model)

#     plt.figure()
#     plt.rcParams.update({"font.size": 12})
#     plt.title(f'Utterance-level attack ROC Curve - {args.model}')
#     plt.plot(FPRs, TPRs, color="darkorange", lw=2, label=f"ROC curve (area = {avg_AUC:0.2f})")
#     plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.legend(loc="lower right")
#     plt.savefig(
#         os.path.join(args.output_path, f"{args.model}-utterance-level-attack-ROC-curve.png")
#     )

#     df = pd.DataFrame(
#         {
#             "Seen_uttr_sim": seen_uttr_sim, 
#             "Unseen_uttr_sim": unseen_uttr_sim
#         }
#     )
#     df.to_csv(
#         os.path.join(args.output_path, f"{args.model}-utterance-level-attack-similarity.csv"),
#         index=False,
#     )


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

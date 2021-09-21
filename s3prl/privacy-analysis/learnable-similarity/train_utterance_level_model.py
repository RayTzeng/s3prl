import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import IPython

from dataset.dataset import UtteranceLevelDataset, CertainUtteranceDataset
from model.learnable_similarity_model import UtteranceLevelModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed()
    AUXILIARY_DATA_SIZE = args.auxiliary_data_choice_size

    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]

    if args.utterance_list is not None:
        print("Using utterance list....")
        df = pd.read_csv(args.utterance_list, index_col=False)
        utterances = df['Unseen_uttr'].values
        similarity = df['Similarity'].values
        sorted_similarity, sorted_utterances = zip(*sorted(zip(similarity, utterances)))
        sorted_similarity = list(sorted_similarity)
        sorted_utterances = list(sorted_utterances)
        # IPython.embed()
        negative_utterances = sorted_utterances[-AUXILIARY_DATA_SIZE:]
        positive_utterances = sorted_utterances[:AUXILIARY_DATA_SIZE]
        train_dataset = CertainUtteranceDataset(
            args.base_path, positive_utterances, negative_utterances, args.model
        )

        eval_negative_utterances = sorted_utterances[-2*AUXILIARY_DATA_SIZE:-AUXILIARY_DATA_SIZE]
        eval_positive_utterances = sorted_utterances[AUXILIARY_DATA_SIZE:2*AUXILIARY_DATA_SIZE]
        eval_dataset = CertainUtteranceDataset(
            args.base_path, eval_positive_utterances, eval_negative_utterances, args.model
        )

    else:
        train_dataset = UtteranceLevelDataset(
            args.base_path, seen_splits, unseen_splits, AUXILIARY_DATA_SIZE, args.model
        )
        eval_dataset = UtteranceLevelDataset(
            args.base_path, seen_splits, unseen_splits, AUXILIARY_DATA_SIZE, args.model
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=eval_dataset.collate_fn,
    )

    feature, _ = train_dataset[0]
    input_dim = feature.shape[-1]
    # input_dim = 768
    print(f"input dimension: {input_dim}")

    model = UtteranceLevelModel(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    min_loss = 1000
    early_stopping = 0
    epoch = 0
    while epoch < args.n_epochs:
        model.train()
        for batch_id, (features, labels) in enumerate(
            tqdm(train_dataloader, dynamic_ncols=True, desc=f"Train | Epoch {epoch+1}")
        ):
            features = [
                torch.FloatTensor(feature).to(device) for feature in features
            ]
            labels = torch.FloatTensor([label for label in labels]).to(device)
            pred = model(features)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            # if batch_id % 25 == 0:
            #     str_pred = ",".join(
            #         [f"{x:.4f}" for x in pred.detach().cpu().tolist()[:10]]
            #     )
            #     str_label = ",".join(
            #         [f"{x:.4f}" for x in labels.detach().cpu().tolist()[:10]]
            #     )
            #     tqdm.write(f"[Train] Loss :      {loss.detach().cpu().item()}")
            #     tqdm.write(f"[Train] Prediction: {str_pred}")
            #     tqdm.write(f"[Train] Label :     {str_label}")
            #     tqdm.write(" ")

        model.eval()
        total_loss = []
        for batch_id, (features, labels) in enumerate(
            tqdm(eval_dataloader, dynamic_ncols=True, desc="Eval")
        ):
            features = [
                torch.FloatTensor(feature).to(device) for feature in features
            ]
            labels = torch.FloatTensor([label for label in labels]).to(device)
            with torch.no_grad():
                pred = model(features)

            loss = criterion(pred, labels)
            total_loss.append(loss.detach().cpu().item())

            # if batch_id % 10 == 0:
            #     str_pred = ",".join(
            #         [f"{x:.4f}" for x in pred.detach().cpu().tolist()[:10]]
            #     )
            #     str_label = ",".join(
            #         [f"{x:.4f}" for x in labels.detach().cpu().tolist()[:10]]
            #     )
            #     tqdm.write(f"[Eval] Loss :      {loss.detach().cpu().item()}")
            #     tqdm.write(f"[Eval] Prediction: {str_pred}")
            #     tqdm.write(f"[Eval] Label:      {str_label}")
            #     tqdm.write(" ")
        total_loss = np.mean(total_loss)

        if total_loss < min_loss:
            min_loss = total_loss
            print(f"Saving model (epoch = {(epoch + 1):4d}, loss = {min_loss:.4f})")
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.output_path, f"learnable-similarity-{args.model}.pt",
                ),
            )
            early_stopping = 0
        else:
            print(f"Not saving model (epoch = {(epoch + 1):4d}, loss = {total_loss:.4f})")
            early_stopping = early_stopping + 1

        if early_stopping < 5:
            epoch = epoch + 1
        else:
            epoch = args.n_epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", help="directory of feature of LibriSpeech dataset"
    )
    parser.add_argument("--output_path", help="directory to save the model")
    parser.add_argument(
        "--model", help="which self-supervised model you used to extract features"
    )
    # parser.add_argument("--seed", type=int, default=57, help="random seed")
    parser.add_argument(
        "--auxiliary_data_choice_size", type=int, default=1000, help="how many utterance to pick"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="evaluation batch size"
    )
    parser.add_argument(
        "--utterance_list", type=str, default=None, help="certain utterance list"
    )
    parser.add_argument("--n_epochs", type=int, default=30, help="training epoch")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    main(args)

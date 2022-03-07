import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import pdb

from .dataset import VoiceMOSDataset
from .model import Model

warnings.filterwarnings("ignore")


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = kwargs["expdir"]
        idtable = Path(kwargs["expdir"]) / "idtable.pkl"

        self.train_dataset = VoiceMOSDataset(
            load_file(self.datarc["voiceMOS_path"], "sets/TRAINSET"),
            load_file(self.datarc["voiceMOS_path"], "sets/train_mos_list.txt"),
            self.datarc["voiceMOS_path"],
            idtable=idtable,
        )
        self.dev_dataset = VoiceMOSDataset(
            load_file(self.datarc["voiceMOS_path"], "sets/DEVSET"),
            load_file(self.datarc["voiceMOS_path"], "sets/val_mos_list.txt"),
            self.datarc["voiceMOS_path"],
            idtable=idtable,
            valid=True,
        )
        self.test_dataset = VoiceMOSDataset(
            load_file(self.datarc["voiceMOS_path"], "sets/DEVSET"),
            load_file(self.datarc["voiceMOS_path"], "sets/val_mos_list.txt"),
            self.datarc["voiceMOS_path"],
            idtable=idtable,
            valid=True,
        )

        self.system_mos = pd.read_csv(
            Path(
                self.datarc["voiceMOS_path"],
                "system_level_mos.csv",
            ),
            index_col=False
        )


        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])
        self.model = Model(
            input_dim=self.modelrc["projector_dim"],
            clipping=self.modelrc["clipping"] if "clipping" in self.modelrc else False,
            attention_pooling=self.modelrc["attention_pooling"]
            if "attention_pooling" in self.modelrc
            else False,
            num_judges=5000,
        )
        self.objective = nn.MSELoss()
        self.segment_weight = self.modelrc["segment_weight"]
        self.bias_weight = self.modelrc["bias_weight"]

        self.best_scores = {
            "dev_MSE": np.inf,
            "dev_LCC": -np.inf,
            "dev_SRCC": -np.inf,
        }

    # Interface
    def get_dataloader(self, mode):
        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev" or mode == "test":
            return self._get_eval_dataloader(self.dev_dataset)



    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc["train_batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def forward(
        self,
        mode,
        features,
        system_name_list,
        prefix_sums,
        opinion_score_list,
        mos_list,
        segment_judge_ids,
        wav_name_list,
        records,
        **kwargs,
    ):

        features = torch.stack(features)
        features = self.connector(features)

        uttr_scores = []
        bias_scores = []
        if mode == "train":
            mos_list = mos_list.to(features.device)
            segment_judge_ids = segment_judge_ids.to(features.device)
            opinion_score_list = opinion_score_list.to(features.device)
            segments_scores, segments_bias_scores = self.model(
                features, judge_ids=segment_judge_ids
            )
            segments_loss = 0
            uttr_loss = 0
            bias_loss = 0
            for i in range(len(prefix_sums) - 1):
                current_segment_scores = segments_scores[
                    prefix_sums[i] : prefix_sums[i + 1]
                ]
                current_bias_scores = segments_bias_scores[
                    prefix_sums[i] : prefix_sums[i + 1]
                ]
                uttr_score = current_segment_scores.mean(dim=-1)
                uttr_scores.append(uttr_score.detach().cpu())
                bias_score = current_bias_scores.mean(dim=-1)
                bias_scores.append(bias_score.detach().cpu())
                segments_loss += self.objective(current_segment_scores, mos_list[i])
                uttr_loss += self.objective(uttr_score, mos_list[i])
                bias_loss += self.objective(bias_score, opinion_score_list[i])
            segments_loss /= len(prefix_sums) - 1
            uttr_loss /= len(prefix_sums) - 1
            bias_loss /= len(prefix_sums) - 1
            loss = (
                self.segment_weight * segments_loss
                + self.bias_weight * bias_loss
                + uttr_loss
            )

            # for i in range(5):
            #     print(uttr_scores[i], bias_scores[i])

            records["segment loss"].append(segments_loss.item())
            records["utterance loss"].append(uttr_loss.item())
            records["bias loss"].append(bias_loss.item())
            records["total loss"].append(loss.item())

            records["pred_scores"] += uttr_scores
            records["true_scores"] += mos_list.detach().cpu().tolist()

        if mode == "dev":
            mos_list = mos_list.to(features.device)
            segments_scores = self.model(features)
            segments_loss = 0
            uttr_loss = 0
            for i in range(len(prefix_sums) - 1):
                current_segment_scores = segments_scores[
                    prefix_sums[i] : prefix_sums[i + 1]
                ]
                uttr_score = current_segment_scores.mean(dim=-1)
                uttr_scores.append(uttr_score.detach().cpu())
                segments_loss += self.objective(current_segment_scores, mos_list[i])
                uttr_loss += self.objective(uttr_score, mos_list[i])
            segments_loss /= len(prefix_sums) - 1
            uttr_loss /= len(prefix_sums) - 1
            loss = segments_loss + uttr_loss

            records["total loss"].append(loss.item())

            records["pred_scores"] += uttr_scores
            records["true_scores"] += mos_list.detach().cpu().tolist()

            records["wav_names"] += wav_name_list

        if mode == "test":
            segments_scores = self.model(features)

            for i in range(len(prefix_sums) - 1):
                current_segment_scores = segments_scores[
                    prefix_sums[i] : prefix_sums[i + 1]
                ]
                uttr_score = current_segment_scores.mean(dim=-1)
                uttr_scores.append(uttr_score.detach().cpu())

            records["pred_scores"] += uttr_scores
            records["true_scores"] += mos_list.detach().cpu().tolist()

            records["wav_names"] += wav_name_list
        


        if len(records["system"]) == 0:
            records["system"].append(defaultdict(list))
        for i in range(len(system_name_list)):
            records["system"][0][system_name_list[i]].append(uttr_scores[i].tolist())

        if mode == "train":
            return loss

        return 0

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        save_names = []

        # logging loss

        if mode == "train":
            avg_uttr_loss = torch.FloatTensor(records["utterance loss"]).mean().item()
            avg_frame_loss = torch.FloatTensor(records["segment loss"]).mean().item()
            avg_bias_loss = torch.FloatTensor(records["bias loss"]).mean().item()

            logger.add_scalar(
                f"voiceMOS/{mode}-utterance loss",
                avg_uttr_loss,
                global_step=global_step,
            )
            logger.add_scalar(
                f"voiceMOS/{mode}-segment loss",
                avg_frame_loss,
                global_step=global_step,
            )
            logger.add_scalar(
                f"voiceMOS/{mode}-bias loss",
                avg_bias_loss,
                global_step=global_step,
            )

        avg_total_loss = torch.FloatTensor(records["total loss"]).mean().item()

        logger.add_scalar(
            f"voiceMOS/{mode}-total loss",
            avg_total_loss,
            global_step=global_step,
        )

        # logging Utterance-level MSE, LCC, SRCC

        if mode == "dev":
            # some evaluation-only processing, eg. decoding
            all_pred_scores = records["pred_scores"]
            all_true_scores = records["true_scores"]
            all_pred_scores = np.array(all_pred_scores)
            all_true_scores = np.array(all_true_scores)
            MSE = np.mean((all_true_scores - all_pred_scores) ** 2)
            logger.add_scalar(
                f"voiceMOS/{mode}-Utterance level MSE",
                MSE,
                global_step=global_step,
            )
            pearson_rho, _ = pearsonr(all_true_scores, all_pred_scores)
            logger.add_scalar(
                f"voiceMOS/{mode}-Utterance level LCC",
                pearson_rho,
                global_step=global_step,
            )
            spearman_rho, _ = spearmanr(all_true_scores.T, all_pred_scores.T)
            logger.add_scalar(
                f"voiceMOS/{mode}-Utterance level SRCC",
                spearman_rho,
                global_step=global_step,
            )

            tqdm.write(f"[{mode}] Utterance-level MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] Utterance-level LCC  = {pearson_rho:.4f}")
            tqdm.write(f"[{mode}] Utterance-level SRCC = {spearman_rho:.4f}")

            # select which system mos to use
            system_level_mos = self.system_mos

            # logging System-level MSE, LCC, SRCC
            all_system_pred_scores = []
            all_system_true_scores = []

            for key, values in records["system"][0].items():
                all_system_pred_scores.append(np.mean(values))
                all_system_true_scores.append(system_level_mos[system_level_mos['system_ID']==key]['mean'].values[0])

            all_system_pred_scores = np.array(all_system_pred_scores)
            all_system_true_scores = np.array(all_system_true_scores)

            MSE = np.mean((all_system_true_scores - all_system_pred_scores) ** 2)
            pearson_rho, _ = pearsonr(all_system_true_scores, all_system_pred_scores)
            spearman_rho, _ = spearmanr(all_system_true_scores, all_system_pred_scores)

            tqdm.write(f"[{mode}] System-level MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] System-level LCC  = {pearson_rho:.4f}")
            tqdm.write(f"[{mode}] System-level SRCC = {spearman_rho:.4f}")

            logger.add_scalar(
                f"voiceMOS/{mode}-System level MSE",
                MSE,
                global_step=global_step,
            )
            logger.add_scalar(
                f"voiceMOS/{mode}-System level LCC",
                pearson_rho,
                global_step=global_step,
            )
            logger.add_scalar(
                f"voiceMOS/{mode}-System level SRCC",
                spearman_rho,
                global_step=global_step,
            )

        # save model
        if mode == "dev":
            if avg_total_loss < self.best_scores["dev_MSE"]:
                self.best_scores["dev_MSE"] = avg_total_loss
                save_names.append(f"{mode}-MSE-best.ckpt")
                df = pd.DataFrame(list(zip(records["wav_names"], np.array(records["pred_scores"]))))
                df.to_csv(Path(self.expdir, "MSE-best-answer.txt"), header=None, index=None)
                tqdm.write(f"writing answer.txt")

            if pearson_rho > self.best_scores["dev_LCC"]:
                self.best_scores["dev_LCC"] = pearson_rho
                save_names.append(f"{mode}-LCC-best.ckpt")
                df = pd.DataFrame(list(zip(records["wav_names"], np.array(records["pred_scores"]))))
                df.to_csv(Path(self.expdir, "LCC-best-answer.txt"), header=None, index=None)
                tqdm.write(f"writing answer.txt")

            if spearman_rho > self.best_scores["dev_SRCC"]:
                self.best_scores["dev_SRCC"] = spearman_rho
                save_names.append(f"{mode}-SRCC-best.ckpt")
                df = pd.DataFrame(list(zip(records["wav_names"], np.array(records["pred_scores"]))))
                df.to_csv(Path(self.expdir, "SRCC-best-answer.txt"), header=None, index=None)
                tqdm.write(f"writing answer.txt")

        if mode == "test":
            df = pd.DataFrame(list(zip(records["wav_names"], np.array(records["pred_scores"]))))
            df.to_csv(Path(self.expdir, "answer.txt"), header=None, index=None)
            


        return save_names


def load_file(base_path, file):
    dataframe = pd.read_csv(Path(base_path, file), header=None)
    return dataframe

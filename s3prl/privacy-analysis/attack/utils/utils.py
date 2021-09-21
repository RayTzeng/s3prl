import numpy as np
import math
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import torch
import torchaudio
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import Spectrogram, MelScale, MFCC
from torchaudio.functional import magphase, compute_deltas
import yaml



############
# CONSTANT #
############
N_SAMPLED_PSEUDO_WAV = 2
SAMPLE_RATE = 16000


def compute_speaker_adversarial_advantage_by_percentile(seen_sim, unseen_sim, percentile_choice, model=""):
    adversarial_advantage_by_percentile = []
    true_positive_rate_by_percentile = []
    false_positive_rate_by_percentile = []

    for percentile in percentile_choice:
        sorted_unseen_sim = sorted(unseen_sim)
        threshold = sorted_unseen_sim[math.floor(len(unseen_sim) * percentile / 100)]
        TP = len([sim for sim in seen_sim if sim >= threshold])
        FN = len([sim for sim in seen_sim if sim < threshold])
        FP = len([sim for sim in unseen_sim if sim >= threshold])
        TN = len([sim for sim in unseen_sim if sim < threshold])

        true_positive_rate = TP / (TP + FN)
        false_positive_rate = FP / (TN + FP)
        adversarial_advantage = true_positive_rate - false_positive_rate

        true_positive_rate_by_percentile.append(true_positive_rate)
        false_positive_rate_by_percentile.append(false_positive_rate)
        adversarial_advantage_by_percentile.append(adversarial_advantage)


    print()
    print(f"[{model}]")
    print("Percentile           :", " | ".join(f"{num:5}%" for num in percentile_choice))
    print("-----------------------------------------------------------------")
    print("True Positive Rate   :", " | ".join(f"{num:.4f}" for num in true_positive_rate_by_percentile))
    print("False Positive Rate  :", " | ".join(f"{num:.4f}" for num in false_positive_rate_by_percentile))
    print("Adversarial Advantage:", " | ".join(f"{num:.4f}" for num in adversarial_advantage_by_percentile))
    print()

    return true_positive_rate_by_percentile, false_positive_rate_by_percentile, adversarial_advantage_by_percentile

def compute_speaker_adversarial_advantage_by_ROC(seen_sim, unseen_sim, model=""):
    average_threshold = np.mean(unseen_sim)
    all_sim = np.concatenate((seen_sim, unseen_sim))
    prediction = (all_sim >= average_threshold)
    true_label = np.concatenate((np.ones_like(seen_sim), np.zeros_like(unseen_sim)))

    average_AUC = roc_auc_score(true_label, all_sim)
    print(f"[{model}]")
    print(f'Average membership AUC: {average_AUC:.4f}')

    # Adversarial advantage by average threshold
    TN, FP, FN, TP = confusion_matrix(prediction, true_label).ravel()
    true_positive_rate = TP / (TP + FN)
    false_positive_rate = FP / (TN + FP)
    average_adversarial_advantage = true_positive_rate - false_positive_rate
    print(f'Average adversarial advantage: {average_adversarial_advantage:.4f} with threshold {average_threshold:.4f}')

    # Adversarial advantage by best threshold
    false_positive_rates, true_positive_rates, thresholds = roc_curve(true_label, all_sim)
    adversarial_advantages = true_positive_rates - false_positive_rates
    best_idx = np.argmax(adversarial_advantages)
    best_adversarial_advantage = adversarial_advantages[best_idx]
    best_threshold = thresholds[best_idx]
    print(f'Best adversarial advantage: {best_adversarial_advantage:.4f} with threshold {best_threshold:.4f}')
    print()

    return true_positive_rates, false_positive_rates, average_AUC

def compute_utterance_adversarial_advantage_by_percentile(seen_sim, unseen_sim, percentile_choice, model=""):
    seen_sim = 1 - np.array(seen_sim)
    unseen_sim = 1 - np.array(unseen_sim)
    adversarial_advantage_by_percentile = []
    true_positive_rate_by_percentile = []
    false_positive_rate_by_percentile = []

    for percentile in percentile_choice:
        sorted_unseen_sim = sorted(unseen_sim)
        threshold = sorted_unseen_sim[math.floor(len(unseen_sim) * (100-percentile) / 100)]
        TP = len([sim for sim in seen_sim if sim >= threshold])
        FN = len([sim for sim in seen_sim if sim < threshold])
        FP = len([sim for sim in unseen_sim if sim >= threshold])
        TN = len([sim for sim in unseen_sim if sim < threshold])

        true_positive_rate = TP / (TP + FN)
        false_positive_rate = FP / (TN + FP)
        adversarial_advantage = true_positive_rate - false_positive_rate

        true_positive_rate_by_percentile.append(true_positive_rate)
        false_positive_rate_by_percentile.append(false_positive_rate)
        adversarial_advantage_by_percentile.append(adversarial_advantage)


    print()
    print(f"[{model}]")
    print("Percentile           :", " | ".join(f"{num:5}%" for num in percentile_choice))
    print("-----------------------------------------------------------------")
    print("True Positive Rate   :", " | ".join(f"{num:.4f}" for num in true_positive_rate_by_percentile))
    print("False Positive Rate  :", " | ".join(f"{num:.4f}" for num in false_positive_rate_by_percentile))
    print("Adversarial Advantage:", " | ".join(f"{num:.4f}" for num in adversarial_advantage_by_percentile))
    print()

    return true_positive_rate_by_percentile, false_positive_rate_by_percentile, adversarial_advantage_by_percentile

def compute_utterance_adversarial_advantage_by_ROC(seen_sim, unseen_sim, model=""):
    seen_sim = 1 - np.array(seen_sim)
    unseen_sim = 1 - np.array(unseen_sim)
    average_threshold = np.mean(unseen_sim)
    all_sim = np.concatenate((seen_sim, unseen_sim))
    prediction = (all_sim >= average_threshold)
    true_label = np.concatenate((np.ones_like(seen_sim), np.zeros_like(unseen_sim)))

    average_AUC = roc_auc_score(true_label, all_sim)
    print(f"[{model}]")
    print(f'Average membership AUC: {average_AUC:.4f}')

    # Adversarial advantage by average threshold
    TP = len([sim for sim in seen_sim if sim >= average_threshold])
    FN = len([sim for sim in seen_sim if sim < average_threshold])
    FP = len([sim for sim in unseen_sim if sim >= average_threshold])
    TN = len([sim for sim in unseen_sim if sim < average_threshold])
    true_positive_rate = TP / (TP + FN)
    false_positive_rate = FP / (TN + FP)
    average_adversarial_advantage = true_positive_rate - false_positive_rate
    print(f'Average adversarial advantage: {average_adversarial_advantage:.4f} with threshold {average_threshold:.4f}')

    # Adversarial advantage by best threshold
    false_positive_rates, true_positive_rates, thresholds = roc_curve(true_label, all_sim)
    adversarial_advantages = true_positive_rates - false_positive_rates
    best_idx = np.argmax(adversarial_advantages)
    best_adversarial_advantage = adversarial_advantages[best_idx]
    best_threshold = thresholds[best_idx]
    print(f'Best adversarial advantage: {best_adversarial_advantage:.4f} with threshold {best_threshold:.4f}')
    print()

    return true_positive_rates, false_positive_rates, average_AUC

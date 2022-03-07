from pathlib import Path
import os

import random
import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate

class VoiceMOSDataset(Dataset):
    def __init__(self, dataframe, mos_list, base_path, segments_duration=1, idtable = '', valid = False):
        self.base_path = Path(base_path)
        self.dataframe = dataframe
        self.mos_list = mos_list
        self.segments_durations = segments_duration
        self.valid=valid
        self._JUDGE=4

        if not self.valid:
            if Path.is_file(idtable):
                self.idtable = torch.load(idtable)
                for i, judge_i in enumerate(self.dataframe[self._JUDGE]):
                    self.dataframe[self._JUDGE][i] = self.idtable[judge_i]
            else:
                self.gen_idtable(idtable) 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        system_name, wav_name, opinion_score, _, judge_id = self.dataframe.loc[idx]
        mos = self.mos_list[self.mos_list[0]==wav_name][1].values[0]
        wav_path = self.base_path / "wav" / wav_name
        wav, _ = apply_effects_file(
            str(wav_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ],
        )

        wav = wav.view(-1)
        wav_segments = unfold_segments(wav, self.segments_durations)

        return system_name, wav_segments, opinion_score, mos, judge_id, wav_name

    def collate_fn(self, samples):
        system_name_list, wav_segments_list, opinion_score_list, mos_list, judge_id_list, wav_name_list = zip(*samples)
        flattened_wavs_segments = [
            wav_segment
            for wav_segments in wav_segments_list
            for wav_segment in wav_segments
        ]
        wav_segments_lengths = [len(wav_segments) for wav_segments in wav_segments_list]
        prefix_sums = list(accumulate(wav_segments_lengths, initial=0))

        if self.valid:
                return (
                torch.stack(flattened_wavs_segments),
                system_name_list,
                prefix_sums,
                torch.FloatTensor(opinion_score_list),
                torch.FloatTensor(mos_list),
                None,
                wav_name_list
            )
            
        segment_judge_ids = []
        for i in range(len(prefix_sums)-1):
            segment_judge_ids.extend([judge_id_list[i]] * (prefix_sums[i+1]-prefix_sums[i]))

        return (
            torch.stack(flattened_wavs_segments),
            system_name_list,
            prefix_sums,
            torch.FloatTensor(opinion_score_list),
            torch.FloatTensor(mos_list),
            torch.LongTensor(segment_judge_ids),
            wav_name_list
        )
    
    def gen_idtable(self, idtable_path):
        if idtable_path == '':
            idtable_path = './idtable.pkl'
        self.idtable = {}
        count = 0
        for i, judge_i in enumerate(self.dataframe[self._JUDGE]):
            if judge_i not in self.idtable.keys():
                self.idtable[judge_i] = count
                count += 1
                self.dataframe[self._JUDGE][i] = self.idtable[judge_i]
            else:
                self.dataframe[self._JUDGE][i] = self.idtable[judge_i]
        torch.save(self.idtable, idtable_path)


def unfold_segments(tensor, tgt_duration, sample_rate=16000):
    seg_lengths = int(tgt_duration * sample_rate)
    src_lengths = len(tensor)
    step = seg_lengths // 2
    tgt_lengths = (
        seg_lengths if src_lengths <= seg_lengths else (src_lengths // step + 1) * step
    )

    pad_lengths = tgt_lengths - src_lengths
    padded_tensor = torch.cat([tensor, torch.zeros(pad_lengths)])
    segments = padded_tensor.unfold(0, seg_lengths, step).unbind(0)

    return segments
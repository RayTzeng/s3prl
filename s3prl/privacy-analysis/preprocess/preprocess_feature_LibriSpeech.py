import os
import glob
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torchaudio.sox_effects import apply_effects_file
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from s3prl.upstream.tera.expert import UpstreamExpert
import IPython

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    if not args.output_name:
        args.output_name = args.model
    
    if args.state_dict:
        if 'tera' in args.model:
            options = {
                "load_pretrain": "True",
                "no_grad": "False",
                "dropout": "default",
                "spec_aug": "False",
                "spec_aug_prev": "True",
                "output_hidden_states": "True",
                "permute_input": "False",
            }
            model = UpstreamExpert(ckpt = args.state_dict)
    else:
        model = torch.hub.load('s3prl/s3prl', args.model).to(device)
    model = model.to(device)
    model.eval()

    split_path = os.path.join(args.base_path, args.split)
    
    speaker_count = 0
    audio_count = 0
    for speaker in tqdm(glob.glob(os.path.join(split_path, '*[!.txt]')), ascii=True, desc="Speaker"):
        speaker_count += 1
        for chapter in glob.glob(os.path.join(split_path, speaker, '*')):

            # check if output folder exist
            output_folder = os.path.join(args.output_path, args.split, speaker.split("/")[-1], chapter.split("/")[-1])
            # print(args.output_path)
            # print(output_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                tqdm.write(f"Directory {output_folder} Created ")

            # 
            for audio_path in glob.glob(os.path.join(split_path, speaker, chapter,'*.flac')):
                audio_count += 1

                audio_name = audio_path.split("/")[-1]

                wav, _ = apply_effects_file(
                    audio_path,
                    [
                        ["channels", "1"],
                        ["rate", "16000"],
                        ["norm"],
                    ],
                )
                wav = wav.squeeze(0).to(device)
                with torch.no_grad():
                    feature = model([wav])['last_hidden_state']

                output_path = os.path.join(args.output_path, args.split, speaker.split("/")[-1], chapter.split("/")[-1], f"{args.output_name}-{audio_name}.pt")
                tqdm.write(output_path)
                torch.save(feature.cpu(), output_path)
                
                

    print("There are {} speakers".format(speaker_count))
    print("There are {} audios".format(audio_count))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", help="directory of LibriSpeech dataset")
    parser.add_argument("--split", help="which split of LibriSpeech")
    parser.add_argument("--state_dict", help='pre-trained state dict path')
    parser.add_argument("--model_cfg", help = "pre-trained model config path")
    parser.add_argument("--output_path", help="directory to save feautures")
    parser.add_argument("--output_name", help="filename to save")
    parser.add_argument("--model", help="which self-supervised model to extract features")
    args = parser.parse_args()

    

    main(args)

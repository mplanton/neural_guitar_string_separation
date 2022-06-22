# Load trained model and make source separation with CREPE f0-tracks of a whole
# song.

import torch
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import os
import tqdm
import argparse
import json
import glob
import random

import models # Needed to load trained model architecture
import utils
from data import CSDSongDataset as CSDSongDataset
from data import Guitarset as Guitarset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, help="Model tag defined in the config at training time.")
    parser.add_argument('--test-set', type=str, choices=['CSD', 'Guitarset'], help="Dataset used for inference.")
    parser.add_argument('--song-name', type=str, help="Song name of the specified dataset used for inference.")
    args, _ = parser.parse_known_args()
    tag = args.tag

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load trained model
    trained_model, model_args = utils.load_model(tag, device, return_args=True)
    trained_model.return_synth_params = False
    trained_model.return_sources = True

    batch_size = model_args['batch_size']
    n_samples = model_args['example_length']
    n_sources = model_args['n_sources']
    
    if args.test_set == "CSD":
        loader = iter(DataLoader(CSDSongDataset(model_args=model_args, song_name=args.song_name),
                                                batch_size=batch_size, shuffle=False))
    elif args.test_set == "Guitarset":
        # Get files from args
        ds_path = "../Datasets/Guitarset/audio_16kHz/"
        song_name = args.song_name.split('.')[0] # Get rid of eventual file extension.
        file_list = sorted(glob.glob(ds_path + song_name + ".wav"))
        
        loader = iter(DataLoader(
            Guitarset(
                dataset_range=(0, 0), # not used for inference
                style=model_args['style'],
                genres=model_args['genres'],
                allowed_strings=model_args['strings'],
                shuffle_files=False,
                conf_threshold=model_args['confidence_threshold'],
                example_length=model_args['example_length'],
                return_name=False,
                f0_from_mix=model_args['f0_cuesta'],
                cunet_original=False,
                file_list=file_list)
        ))
    
    # Feed mix and f0-tracks in example_length chunks to the model for separation
    # and collect the output.
    source_estimates_slices = []
    source_estimates_masking_slices = []

    pbar = tqdm.tqdm(loader)
    for mix_slice, freqs_slice, sources_slice in pbar:
        batch_len = mix_slice.shape[0]
        with torch.no_grad():
            mix_estimate_slice, source_estimates_slice = trained_model(mix_slice, freqs_slice)
        # [batch_size * n_sources, n_samples]
        source_estimates_masking_slice = utils.masking_from_synth_signals_torch(mix_slice, source_estimates_slice, n_fft=2048, n_hop=256)
        source_estimates_masking_slice = torch.reshape(source_estimates_masking_slice, (batch_len, n_sources, n_samples))
        source_estimates_slices.append(source_estimates_slice)
        source_estimates_masking_slices.append(source_estimates_masking_slice)

    # assemble the outputs to continuous singals
    source_estimates_slices = torch.cat(source_estimates_slices).transpose(0, 1)
    source_estimates_masking_slices = torch.cat(source_estimates_masking_slices).transpose(0, 1)
    source_estimates = torch.reshape(source_estimates_slices, (n_sources, -1))
    source_estimates_masking = torch.reshape(source_estimates_masking_slices, (n_sources, -1))

    out_path = "inference/" + tag
    os.makedirs("inference", exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    # Get voice declaration
    if args.test_set == "CSD":
        voice_dict = model_args['voice_dict']
        voice_list = [voice_dict[key] for key in ('b', 'a', 's', 't')]
    elif args.test_set == "Guitarset":
        voice_list = model_args['strings']

    # Save files.
    for i in range(source_estimates.shape[0]):
        voice = str(voice_list[i])
        if args.test_set == "Guitarset":
            voice = "string" + voice
        torchaudio.save(out_path + "/" + args.song_name + "_source_estimate_" + voice + ".wav", source_estimates[i], sample_rate=16000)
        torchaudio.save(out_path + "/" + args.song_name + "_source_estimate_masking_" + voice + ".wav", source_estimates_masking[i], sample_rate=16000)



if __name__ == "__main__":
    main()

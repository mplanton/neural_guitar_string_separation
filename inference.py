# Load trained model and make source separation with CREPE f0-tracks of a whole
# song.

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import tqdm
import argparse
import glob
from scipy.io import wavfile
import json

import matplotlib.pyplot as plt

import models # Needed to load trained model architecture
import utils
from data import CSDSongDataset as CSDSongDataset
from data import Guitarset as Guitarset


def convert_onsets_to_global_inexing(local_onset_list, batch_size, n_sources):
    """
    Convert a list of frame-local onset_frame_indices of shape [n_onsets, 3]
    with indices [batch, string, frame] to global inices.
    """
    frame_counters = np.zeros((batch_size, n_sources))
    
    global_onsets = []
    for onset_frame in local_onset_list:
        for batch, string, frame in onset_frame.numpy():
             frame_counters[batch, string] += frame
             global_onsets.append(
                 np.int_([batch, string, frame_counters[batch, string]]))
    out = np.concatenate(global_onsets)
    out = np.reshape(out, (-1, 3))
    return out
    

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, help="Model tag defined in the config at training time.")
parser.add_argument('--which', type=str, choices=['best', 'last'], help="Load the 'best' or 'last' trained model.")
parser.add_argument('--test-set', type=str, choices=['CSD', 'Guitarset'], help="Dataset used for inference.")
parser.add_argument('--song-names', type=str, nargs='*', help="Song names of the specified dataset used for inference.")
args, _ = parser.parse_known_args()
tag = args.tag

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load trained model
trained_model, model_args = utils.load_model(tag, which=args.which, device=device, return_args=True)
trained_model.return_sources = True

batch_size = model_args['batch_size']
n_samples = model_args['example_length']

if args.test_set == "CSD":
    n_sources = model_args['n_sources']
    trained_model.return_synth_params = False
    loader = iter(DataLoader(CSDSongDataset(model_args=model_args, song_name=args.song_name),
                                            batch_size=batch_size, shuffle=False))
elif args.test_set == "Guitarset":
    print("Warning: Guitarset currently distributes all songs along the batch.")
    print("The songs are saved concatenated as one audio file per batch...")
    n_sources = len(model_args['allowed_strings'])
    # Get files from args
    ds_path = "../Datasets/Guitarset/audio_16kHz/"
    # Get rid of eventual file extension.
    song_names = [song_name.split('.')[0] for song_name in args.song_names]
    # Expand paths
    song_paths = [os.path.join(ds_path + song_name + ".wav") for song_name in song_names]
    file_list = sorted(glob.glob(path) for path in song_paths)
    # Flatten the list
    file_list = [x for xs in file_list for x in xs]

    
    ds = Guitarset(
                batch_size=model_args['batch_size'],
                dataset_range=(0, 0), # not used for inference
                style=model_args['style'],
                genres=model_args['genres'],
                allowed_strings=model_args['allowed_strings'],
                shuffle_files=False,
                conf_threshold=model_args['confidence_threshold'],
                example_length=model_args['example_length'],
                return_name=False,
                f0_from_mix=model_args['f0_cuesta'],
                cunet_original=False,
                file_list=file_list,
                normalize_mix=model_args['normalize_mix'],
                normalize_sources=model_args['normalize_sources'])
    
    # Guitarset does batching on its own.
    loader = DataLoader(ds, batch_size=None, shuffle=False)


# Feed mix and f0-tracks in example_length chunks to the model for separation
# and collect the output.
target_sources_slices = []
source_estimates_slices = []
source_estimates_masking_slices = []

# Synth controls
f0_hz = []
onset_frame_indices = []


trained_model.eval()
pbar = tqdm.tqdm(loader)
for mix_slice, freqs_slice, sources_slice in pbar:
    batch_len = mix_slice.shape[0]
    with torch.no_grad():
        mix_estimate_slice, source_estimates_slice, ctl = \
            trained_model(mix_slice, freqs_slice, return_synth_controls=True)
    #print("DBG: fc:", fc)
    f0_hz.append(ctl['f0_hz'])
    onset_frame_indices.append(ctl['onset_frame_indices'])
    
    # [batch_size * n_sources, n_samples]
    source_estimates_masking_slice = utils.masking_from_synth_signals_torch(mix_slice, source_estimates_slice, n_fft=2048, n_hop=256)
    source_estimates_masking_slice = torch.reshape(source_estimates_masking_slice, (batch_len, n_sources, n_samples))
    source_estimates_slices.append(source_estimates_slice)
    source_estimates_masking_slices.append(source_estimates_masking_slice)
    target_sources_slices.append(torch.transpose(sources_slice, 1, 2))


# assemble the outputs to continuous signals
source_estimates = torch.cat(source_estimates_slices, dim=-1).numpy()
source_estimates_masking = torch.cat(source_estimates_masking_slices, dim=-1).numpy()
target_sources = torch.cat(target_sources_slices, dim=-1).numpy()

f0_hz = torch.cat(f0_hz, dim=-1).numpy()
global_onset_frame_indices = convert_onsets_to_global_inexing(onset_frame_indices, batch_size, n_sources)

out_path = "inference/" + tag + '_' + args.which
os.makedirs(out_path, exist_ok=True)

# Save files.
for batch in range(batch_size):
    n_batch = str(batch).zfill(2)
    wavfile.write(out_path + f"/{n_batch}_source_estimates.wav",
                  rate=16000,
                  data=source_estimates[batch].T)
    wavfile.write(out_path + f"/{n_batch}_source_estimates_masking.wav",
                  rate=16000,
                  data=source_estimates_masking[batch].T)
    wavfile.write(out_path + f"/{n_batch}_target_sources.wav",
                  rate=16000,
                  data=target_sources[batch].T)

np.save(out_path + "/f0_hz.npy", f0_hz) # [batch_size, n_sources, n_frames]
np.save(out_path + "/onset_frame_indices.npy", global_onset_frame_indices) # [n_onsets, 3]

f_name = "inference_songs.json"
with open(os.path.join(out_path, f_name), "w") as file:
    json.dump(file_list, file)

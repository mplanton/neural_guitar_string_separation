# Load trained model and make source separation with CREPE f0-tracks of a whole
# song.

import torch
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import os
import tqdm
import argparse
import glob
from scipy.io import wavfile

import matplotlib.pyplot as plt

import models # Needed to load trained model architecture
import utils
from data import CSDSongDataset as CSDSongDataset
from data import Guitarset as Guitarset


parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, help="Model tag defined in the config at training time.")
parser.add_argument('--test-set', type=str, choices=['CSD', 'Guitarset'], help="Dataset used for inference.")
parser.add_argument('--song-names', type=str, nargs='*', help="Song names of the specified dataset used for inference.")
args, _ = parser.parse_known_args()
tag = args.tag

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load trained model
trained_model, model_args = utils.load_model(tag, device, return_args=True)
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
    print("The songs are saved concatenated as one audio file...")
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
source_estimates_slices = []
source_estimates_masking_slices = []
fcs = []

pbar = tqdm.tqdm(loader)
for mix_slice, freqs_slice, sources_slice in pbar:
    batch_len = mix_slice.shape[0]
    with torch.no_grad():
        mix_estimate_slice, source_estimates_slice, fc = \
            trained_model(mix_slice, freqs_slice, return_fc=True)
    #print("DBG: fc:", fc)
    fcs.append(fc)
    
    # [batch_size * n_sources, n_samples]
    source_estimates_masking_slice = utils.masking_from_synth_signals_torch(mix_slice, source_estimates_slice, n_fft=2048, n_hop=256)
    source_estimates_masking_slice = torch.reshape(source_estimates_masking_slice, (batch_len, n_sources, n_samples))
    source_estimates_slices.append(source_estimates_slice)
    source_estimates_masking_slices.append(source_estimates_masking_slice)

# assemble the outputs to continuous signals
source_estimates = torch.cat(source_estimates_slices, dim=-1).numpy()
source_estimates_masking = torch.cat(source_estimates_masking_slices, dim=-1).numpy()
fcs = torch.cat(fcs, dim=-1).numpy()

out_path = "inference/" + tag
os.makedirs("inference", exist_ok=True)
os.makedirs(out_path, exist_ok=True)

# Get voice declaration
if args.test_set == "CSD":
    voice_dict = {'b':'Bajos', 'a':'ContraAlt', 's':'Soprano', 't':'Tenor'}
    voice_list = [voice_dict[key] for key in ('b', 'a', 's', 't')]
elif args.test_set == "Guitarset":
    voice_list = model_args['allowed_strings']

# Save files.
for batch in range(batch_size):
    wavfile.write(out_path + f"/source_estimates_batch{batch}.wav",
                  rate=16000,
                  data=source_estimates[batch].T)
    wavfile.write(out_path + f"/source_estimates_masking_batch{batch}.wav",
                  rate=16000,
                  data=source_estimates_masking[batch].T)

np.save(out_path + "/fcs.npy", fcs)

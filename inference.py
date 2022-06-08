# Load trained model and make source separation with CREPE f0-tracks.

import torch
from torch.utils.data import Dataset, DataLoader
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


class SongDataset(Dataset):
    """
    A dataset class to make source separation with a single song.
    """
    def __init__(self, model_args, test_set='CSD', song_name="Nino Dios", example_length=64000):
        if test_set == 'CSD':
            self.ds_path = "../Datasets/ChoralSingingDataset/"
        
        # choose sources
        voice_dict = {'b': "Bajos", 'a': "ContraAlt", 's': "Soprano", 't': "Tenor"}
        
        source_paths = []
        for v in model_args['voices']:
            voice = voice_dict[v]
            voice_paths = glob.glob(os.path.join(self.ds_path, song_name, "audio_16kHz",
                                                 "*" + voice + "*.wav"))
            voice_path = random.choice(voice_paths)
            source_paths.append(voice_path)
        
        # load sources
        sources = []
        for source_path in source_paths:
            s, sr = torchaudio.load(source_path)
            sources.append(s)
            self.sr = sr

        # Make them equal length.
        self.example_length = example_length
        lengths = [s.shape[1] for s in sources]
        min_len = min(lengths)
        cut_len = (min_len // example_length) * example_length
        sources = [s[0, :cut_len] for s in sources]

        # load sources f0-tracks
        freqs = []
        for source_path in source_paths:
            f_name = source_path.split('/')[-1].split('.')[0] + "_frequency.npy"
            freq = np.load(os.path.join(self.ds_path, song_name, "crepe_f0_center", f_name))
            freqs.append(freq)

        # Make frequencies equal length
        hopsize = 256 # CREPE hopsize 16ms in samples
        self.hopsize = hopsize
        freqs = [torch.from_numpy(freq[:cut_len // hopsize]) for freq in freqs]
        self.freqs = torch.stack(freqs, dim=1)  # [n_frames, n_sources]
        self.freqs_per_example = example_length // hopsize

        # Make mix from sources and normalize
        sources = torch.stack(sources)  # (n_sources, n_samples)
        mix = torch.sum(sources, dim=0) # (n_samples)
        mix_max = mix.abs().max()
        self.mix = mix / mix_max

    
    def __len__(self):
        return len(self.mix) // self.example_length
    
    def __getitem__(self, index):
        mix_slice = self.mix[index * self.example_length : (index+1) * self.example_length]
        freqs_slice = self.freqs[index * self.freqs_per_example : (index+1) * self.freqs_per_example]
        return mix_slice, freqs_slice

#------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, help="Model tag defined in the config at training time.")
parser.add_argument('--test-set', type=str, choices=['CSD'], help="Dataset used for inference.")
parser.add_argument('--song-name', type=str, default='El Rossinyol', help="Song name of the specified dataset used for inference.")
args, _ = parser.parse_known_args()
tag = args.tag

with open("trained_models/" + tag + "/" + tag + ".json") as f:
    info = json.load(f)

model_path = "trained_models/" + tag

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load trained model
trained_model, model_args = utils.load_model(tag, device, return_args=True)
trained_model.return_synth_params = False
trained_model.return_sources = True

batch_size = model_args['batch_size']
n_samples = model_args['example_length']
n_sources = model_args['n_sources']
loader = iter(DataLoader(SongDataset(model_args=model_args, song_name=args.song_name),
                                     batch_size=batch_size, shuffle=False))

# Feed mix and f0-tracks in example_length chunks to the model for separation
# and collect the output.
source_estimates_slices = []
source_estimates_masking_slices = []

pbar = tqdm.tqdm(loader)
for mix_slice, freqs_slice in pbar:
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

for i in range(source_estimates.shape[0]):
    torchaudio.save(out_path + "/" + args.song_name + "_source_estimate_" + str(i) + ".wav", source_estimates[i], sample_rate=16000)
    torchaudio.save(out_path + "/" + args.song_name + "_source_estimate_masking" + str(i) + ".wav", source_estimates_masking[i], sample_rate=16000)

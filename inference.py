# Load trained model and make source separation with CREPE f0-tracks.

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
import tqdm

import models # trained model architecture
import utils


class SongDataset(Dataset):
    """
    A dataset class to make source separation with a single song.
    """
    def __init__(self, dataset_path="../Datasets/ChoralSingingDataset",
                 song_name="Nino Dios",
                 example_length=64000):
        # load 3 sources
        if song_name == "Nino Dios":
            s1_voice = "nino_Bajos_104"
            s2_voice = "nino_ContraAlt_107"
            s3_voice = "nino_tenor1-06-2"
        elif song_name == "El Rossinyol":
            s1_voice = "rossinyol_Bajos_107"
            s2_voice = "rossinyol_ContraAlt_1-06"
            s3_voice = "rossinyol_Tenor1-09"
        
        s1_path = os.path.join(dataset_path, song_name, "audio_16kHz", s1_voice + ".wav")
        s2_path = os.path.join(dataset_path, song_name, "audio_16kHz", s2_voice + ".wav")
        s3_path = os.path.join(dataset_path, song_name, "audio_16kHz", s3_voice + ".wav")
        s1, sr = torchaudio.load(s1_path)
        s2, sr = torchaudio.load(s2_path)
        s3, sr = torchaudio.load(s3_path)
        self.sr = sr

        # Make them equal length.
        self.example_length = example_length
        min_len = min(s1.shape[1], s2.shape[1], s3.shape[1])
        cut_len = (min_len // example_length) * example_length
        s1 = s1[0,:cut_len]
        s2 = s2[0,:cut_len]
        s3 = s3[0,:cut_len]

        # load sources f0-tracks
        f0_1 = np.load(os.path.join(dataset_path, song_name, "crepe_f0_center", s1_voice + "_frequency.npy"))
        f0_2 = np.load(os.path.join(dataset_path, song_name, "crepe_f0_center", s2_voice + "_frequency.npy"))
        f0_3 = np.load(os.path.join(dataset_path, song_name, "crepe_f0_center", s3_voice + "_frequency.npy"))

        # Make f0s equal length
        hopsize = 256 # CREPE hopsize 16ms in samples
        self.hopsize = hopsize
        f0_1 = torch.from_numpy(f0_1[:cut_len // hopsize])
        f0_2 = torch.from_numpy(f0_2[:cut_len // hopsize])
        f0_3 = torch.from_numpy(f0_3[:cut_len // hopsize])
        self.freqs = torch.stack((f0_1, f0_2, f0_3), dim=1)  # [n_frames, n_sources]
        self.f0s_per_example = example_length // hopsize

        # Make mix from sources and normalize
        mix = s1 + s2 + s3
        mix_max = mix.abs().max()
        self.mix = mix / mix_max

    
    def __len__(self):
        return len(self.mix) // self.example_length
    
    def __getitem__(self, index):
        mix_slice = self.mix[index * self.example_length : (index+1) * self.example_length]
        freqs_slice = self.freqs[index * self.f0s_per_example : (index+1) * self.f0s_per_example]
        return mix_slice, freqs_slice

#------------------------------------------------------------------------------

tag = "small_model"
model_path = "trained_models/small_model"
eval_path = "evaluation"
n_sources = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load trained model
trained_model, model_args = utils.load_model(tag, device, return_args=True)
trained_model.return_synth_params = False
trained_model.return_sources = True

batch_size = model_args['batch_size']
n_samples = model_args['example_length']
loader = iter(DataLoader(SongDataset(song_name="El Rossinyol"), batch_size=batch_size, shuffle=False))

# Feed mix and f0-tracks in example_length chunks to the model for separation
# and collect output.
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

for i in range(source_estimates.shape[0]):
    torchaudio.save("source_estimate_" + str(i) + ".wav", source_estimates[i], sample_rate=16000)
    torchaudio.save("source_estimate_masking" + str(i) + ".wav", source_estimates_masking[i], sample_rate=16000)
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

import synths, core

excitation_length=0.05
n_examples = 2
batch_size = 1
sr = 16000
example_length = 4 # sec
# Number of sources in the mix
J = 2
# The FFT hop size is the audio frame length
fft_hop_size = 256
# Nuber of time samples per train example
M = example_length * sr
# Number of STFT time frames per train example
N = int(M / fft_hop_size)


# Build network output -> synth input

# f0: Play a transposed chord per batch
chord = torch.tensor([60, 64]) # Cmaj9
transposes = torch.arange(batch_size)
midi = []
for transpose in transposes:
    midi.append(chord + transpose)
midi = torch.stack(midi, dim=0).unsqueeze(-1).repeat(1, 1, n_examples*N).unsqueeze(-1)
f0_hz = core.midi_to_hz(midi)

# Add a rhythm by setting f0 to zero
beat_freq = 1.02 # 60bpm
t = torch.linspace(0, n_examples * example_length, n_examples * N)
rhythm_mask = torch.clamp(9999999 * torch.sin(2 * np.pi * beat_freq * t), 0, 1)
rhythm_mask = rhythm_mask.unsqueeze(0).repeat(J, 1).unsqueeze(0).repeat(batch_size, 1, 1)
rhythm_mask = rhythm_mask.unsqueeze(-1)

f0_hz = rhythm_mask * f0_hz
#plt.plot(f0_hz[0, 0].numpy())

# Onsets and offsets are analyzed from f0 by now.
on_offsets = torch.where(f0_hz > 0., torch.tensor(1., device=f0_hz.device),
                                      torch.tensor(0., device=f0_hz.device))
#plt.plot(on_offsets[0, 0].numpy())
#plt.title("on off")

# fc: different for every string
fc = torch.tensor([5000, 5500]).unsqueeze(0)
fc = fc.repeat(batch_size, 1).unsqueeze(-1).repeat(1, 1, n_examples*N).unsqueeze(-1)

# TODO: test time variant parameters!

ks = synths.KarplusStrong(batch_size=batch_size,
                          n_samples=M,
                          sample_rate=sr,
                          audio_frame_size=fft_hop_size,
                          n_strings=J,
                          min_freq=20,
                          excitation_length=excitation_length)

# Synthesize sources from parameters
sources = torch.zeros((batch_size, J, n_examples * M))
for example in range(n_examples):
    print("Calculate example", example)
    f0_in = f0_hz[:, :, example * N : example * N + N]
    fc_in = fc[:, :, example * N : example * N + N]
    on_offsets_in = on_offsets[:, :, example * N : example * N + N]
    controls = ks.get_controls(f0_in,
                               fc_in,
                               on_offsets_in)
    sources[..., example * M : example * M + M] = ks.get_signal(**controls).squeeze(-1)

mix = sources.sum(dim=1)

# Save mix and sources
for batch in range(batch_size):
    # Normalize mix signal
    out_mix = mix[batch]
    factor = max(abs(mix.max().item()), abs(mix.min().item()))
    out_mix_normalized = out_mix / factor
    wavfile.write(f"KS_batch_{batch}_mix.wav", rate=sr, data=out_mix_normalized.numpy())
    for string in range(J):
        wavfile.write(f"KS_batch_{batch}_string_{string}.wav", sr,
                      sources[batch, string].numpy())

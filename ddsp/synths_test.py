import torch
import numpy as np
import scipy.io.wavfile as wavfile

import synths, core


batch_size = 4
sr = 16000

example_length = 4 # sec
# Number of sources in the mix
J = 6

fft_hop_size = 256
# Nuber of time samples per train example
M = example_length * sr
# Number of STFT time frames per train example
N = int(M / fft_hop_size)


# Build network output -> synth input

# f0: Play a transposed chord per batch
chord = torch.tensor([60, 64, 67, 71, 72, 74]) # Cmaj9
transposes = torch.arange(batch_size)
midi = []
for transpose in transposes:
    midi.append(chord + transpose)
midi = torch.stack(midi, dim=0).unsqueeze(-1).repeat(1, 1, N).unsqueeze(-1)
f0_hz = core.midi_to_hz(midi)

# Add a rhythm by setting f0 to zero
beat_freq = 1 # 60bpm
t = torch.linspace(0, example_length, N)
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
fc = torch.tensor([3500, 4000, 4500, 5000, 5500, 6000]).unsqueeze(0)
fc = fc.repeat(batch_size, 1).unsqueeze(-1).repeat(1, 1, N).unsqueeze(-1)

# TODO: test time variant parameters!

ks = synths.KarplusStrong(batch_size=batch_size,
                          n_samples=M,
                          sample_rate=sr,
                          audio_frame_size=N,
                          n_strings=J,
                          min_freq=20)

controls = ks.get_controls(f0_hz, fc, on_offsets)
sources = ks.get_signal(**controls)
mix = sources.sum(dim=1)

# Normalize output signal
out_mix = mix[0].squeeze(-1)
factor = max(abs(mix.max().item()), abs(mix.min().item()))
out_mix_normalized = out_mix / factor

wavfile.write("Karplus_Strong_test_mix.wav", rate=sr, data=out_mix_normalized.numpy())

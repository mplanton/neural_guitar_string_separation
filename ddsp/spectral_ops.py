# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch implementation of DDSP following closely the original code
# https://github.com/magenta/ddsp/blob/master/ddsp/spectral_ops.py

from ddsp.core import safe_log, pad_for_stft
from ddsp.core import torch_float32
#from core import safe_log, pad_for_stft
#from core import torch_float32

#import librosa
import numpy as np
import torch
import torchaudio



def stft(audio, frame_size=2048, overlap=0.75, center=False, pad_end=True):
    """Differentiable stft in PyTorch, computed in batch."""
    audio = torch_float32(audio)
    hop_length = int(frame_size * (1.0 - overlap))
    if pad_end:
        # pad signal so that STFT window is slid until it is
        # completely beyond the signal
        audio = pad_for_stft(audio, frame_size, hop_length)
    assert frame_size * overlap % 2.0 == 0.0
    window = torch.hann_window(int(frame_size), device=audio.device)
    s = torch.stft(
        input=audio,
        n_fft=int(frame_size),
        hop_length=hop_length,
        win_length=int(frame_size),
        window=window,
        center=center)
    return s

def istft(stft, frame_size=2048, overlap=0.75, center=False, length=64000):
    """Differentiable istft in PyTorch, computed in batch."""

    # stft [batch_size, fft_size//2 + 1, n_frames, 2]

    stft = torch_float32(stft)
    hop_length = int(frame_size * (1.0 - overlap))

    assert frame_size * overlap % 2.0 == 0.0
    window = torch.hann_window(int(frame_size), device=stft.device)
    s = torch.istft(
        input=stft,
        n_fft=int(frame_size),
        hop_length=hop_length,
        win_length=int(frame_size),
        window=window,
        center=center,
        length=length)
    return s



def compute_mag(audio, size=2048, overlap=0.75, pad_end=True, center=False, add_in_sqrt=0.0):
    stft_cmplx = stft(audio, frame_size=size, overlap=overlap, center=center, pad_end=pad_end)
    # add_in_sqrt is added before sqrt is taken because the gradient of torch.sqrt(0) is NaN
    mag = torch.sqrt(stft_cmplx[..., 0]**2 + stft_cmplx[..., 1]**2 + add_in_sqrt)
    return torch_float32(mag)

def compute_mel(audio,
                sr=16000,
                lo_hz=20.0,
                hi_hz=8000.0,
                bins=229,
                fft_size=2048,
                overlap=0.75,
                pad_end=True,
                add_in_sqrt=0.0):

    mag = compute_mag(audio, fft_size, overlap, pad_end, center=False, add_in_sqrt=add_in_sqrt)

    mel = torchaudio.transforms.MelScale(n_mels=bins,
                                         sample_rate=sr,
                                         f_min=lo_hz,
                                         f_max=hi_hz).to(mag.device)(mag)
    return mel

def compute_logmel(audio,
                   sr=16000,
                   lo_hz=20.0,
                   hi_hz=8000.0,
                   bins=229,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True,
                   add_in_sqrt=0.0):

    mel = compute_mel(audio, sr, lo_hz, hi_hz, bins, fft_size, overlap, pad_end, add_in_sqrt)
    return safe_log(mel)

def compute_logmag(audio,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True):
    mag = compute_mag(audio, fft_size, overlap, pad_end)
    return safe_log(mag)


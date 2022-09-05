import torch
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

import synths, core

import unittest


def expand_constant(source_constants, n_examples, batch_size, n_frames):
    y = source_constants.unsqueeze(-1).repeat(1, n_frames)
    y = y.unsqueeze(0).repeat(batch_size, 1, 1)
    y = y.unsqueeze(0).repeat(n_examples, 1, 1, 1)
    return y

def generateParameters(frame_rate, batch_size, n_examples, example_length, n_frames, n_sources):
    """
    Build neural network output -> synth input.
    Constant parameters for every string.
    
    Args:
        frame_rate: frame rate of the model in Hz
        batch_size: batch size
        n_examples: number of training examples
        example_length: length of the generated examples in seconds
        n_frames: Number of STFT time frames per train example
        n_sources: Number of sources in the mix
    
    Returns:
        f0_hz: Fundamental frequencies in Hertz,
               torch.Tensor of shape [n_examples, batch_size, n_strings, n_frames]
        fc:    Loop filter cutoff frequency scaled to [0, 1],
               torch.Tensor of shape [n_examples, batch_size, n_strings, n_frames]
        onset_frame_indices: Note onset frame indices to trigger excitation
            signals. One index is [batch, string, onset_frame],
            torch.tensor of shape [n_examples, n_onset_indices, 3]
        fc_ex: Excitation filter cutoff frequency scaled to [0, 1],
               torch.Tensor of shape [n_examples, batch_size, n_strings, n_frames]
        a: Excitation amplitude factor scaled to [0, 1],
               torch.Tensor of shape [n_examples, batch_size, n_strings, n_frames]
    """
    # onset_frame_indices: Note onset frame indices to trigger excitation
    #     signals. One index is [batch, string, onset_frame],
    #     torch.tensor of shape [n_examples, n_onset_indices, 3]
    
    # Make onsets for one example at every quarter second.
    f_onsets = 4 # Hz
    onset_times = torch.arange(0, example_length, 1 / f_onsets)
    onset_frames = (onset_times * frame_rate).type(torch.int) # [n_onset_indices]
    
    n_onset_indices = onset_frames.shape[-1]
    onset_frame_indices = torch.zeros((n_onset_indices, 3))
    for n in range(n_onset_indices):
        batch = n % batch_size
        string = n % n_sources
        onset_frame_indices[n] = torch.tensor([batch, string, onset_frames[n]])
    
    onset_frame_indices = onset_frame_indices.unsqueeze(0).repeat(n_examples, 1, 1)
    onset_frame_indices = onset_frame_indices.type(torch.int)
    
    # f0_hz: Fundamental frequencies in Hertz,
    #        torch.Tensor of shape [n_examples, batch_size, n_strings, n_frames]
    f0_min = 100
    f0_max = 500
    f0s = torch.linspace(f0_min, f0_max, n_sources)
    f0_hz = expand_constant(f0s, n_examples, batch_size, n_frames)
   
    # fc:    Loop filter cutoff frequency scaled to [0, 1],
    #        torch.Tensor of shape [n_examples, batch_size, n_strings, n_frames]
    fc_min = 0.1
    fc_max = 1
    fcs = torch.linspace(fc_min, fc_max, n_sources)
    fc = expand_constant(fcs, n_examples, batch_size, n_frames)
    
    # fc_ex: Excitation filter cutoff frequency scaled to [0, 1],
    #        torch.Tensor of shape [n_examples, batch_size, n_strings, n_frames]
    fc_ex_min = 0.1
    fc_ex_max = 1
    fc_ex = torch.linspace(fc_ex_min, fc_ex_max, n_sources)
    fc_ex = expand_constant(fc_ex, n_examples, batch_size, n_frames)

    # a: Excitation amplitude factor scaled to [0, 1],
    #    torch.Tensor of shape [n_examples, batch_size, n_strings, n_frames]
    a_min = 1
    a_max = 0.1
    a = torch.linspace(a_min, a_max, n_sources)
    a = expand_constant(a, n_examples, batch_size, n_frames)
    
    return f0_hz, fc, onset_frame_indices, fc_ex, a


class TestCore(unittest.TestCase):

    def test_KS_synthetic_input(self):
        save_output=False
        
        excitation_length = 0.005
        excitation_amplitude_scale = 10
        n_examples = 2
        batch_size = 2
        sr = 16000
        #example_length = 2
        example_length = 0.16 # sec
        # Number of sources in the mix
        J = 6
        # The FFT hop size is the audio frame length
        fft_hop_size = 256
        frame_rate = sr // fft_hop_size
        # Nuber of time samples per train example
        M = int(example_length * sr)
        # Number of STFT time frames per train example
        N = int(M / fft_hop_size)
        
        f0_hz, fc, onset_frame_indices, fc_ex, a = \
            generateParameters(frame_rate, batch_size, n_examples, example_length, N, J)
        
        ks = synths.KarplusStrong(batch_size=batch_size,
                                  n_samples=M,
                                  sample_rate=sr,
                                  audio_frame_size=fft_hop_size,
                                  n_strings=J,
                                  min_freq=20,
                                  excitation_length=excitation_length,
                                  excitation_amplitude_scale=excitation_amplitude_scale)
        
        # Synthesize sources from parameters
        sources = torch.zeros((batch_size, J, n_examples * M))
        for example in range(n_examples):
            #print("Calculate example", example)
            
            f0_in = f0_hz[example]
            fc_in = fc[example]
            onset_frame_indices_in = onset_frame_indices[example]
            fc_ex_in = fc_ex[example]
            a_in = a[example]
            
            controls = ks.get_controls(f0_in,
                                       fc_in,
                                       onset_frame_indices_in,
                                       fc_ex_in,
                                       a_in)
            sources[..., example * M : example * M + M] = ks.get_signal(**controls).squeeze(-1)
        mix = sources.sum(dim=1)
        
        # Save mix and sources
        if save_output == True:
            for batch in range(batch_size):
                # Normalize mix signal
                out_mix = mix[batch]
                factor = max(abs(mix.max().item()), abs(mix.min().item()))
                out_mix_normalized = out_mix / factor
                wavfile.write(f"KSsynth_batch_{batch}_mix.wav", rate=sr, data=out_mix_normalized.numpy())
                for string in range(J):
                    wavfile.write(f"KSsynth_batch_{batch}_string_{string}.wav", sr,
                                  sources[batch, string].numpy())
    
    
    def test_KS_differentiability(self):
        excitation_length=0.005
        excitation_amplitude_scale = 10
        n_examples = 2
        batch_size = 2
        sr = 16000
        example_length = 0.16 # sec
        # Number of sources in the mix
        J = 2
        # The FFT hop size is the audio frame length
        fft_hop_size = 256
        frame_rate = sr // fft_hop_size
        # Nuber of time samples per train example
        M = int(example_length * sr)
        # Number of STFT time frames per train example
        N = int(M / fft_hop_size)
        
        f0_hz, fc, onset_frame_indices, fc_ex, a = \
            generateParameters(frame_rate, batch_size, n_examples, example_length, N, J)
        
        ks = synths.KarplusStrong(batch_size=batch_size,
                                  n_samples=M,
                                  sample_rate=sr,
                                  audio_frame_size=fft_hop_size,
                                  n_strings=J,
                                  min_freq=20,
                                  excitation_length=excitation_length,
                                  excitation_amplitude_scale=excitation_amplitude_scale)
        
        for example in range(n_examples):
            #print("Calculate example", example)
            f0_in = f0_hz[example]
            fc_in = fc[example]
            onset_frame_indices_in = onset_frame_indices[example]
            fc_ex_in = fc_ex[example]
            a_in = a[example]
            
            #check_attributes_for_gradients(ks)

            # Zeroing out the gradient
            if fc_in.grad is not None:
                fc_in.grad.zero_()
            if fc_ex_in.grad is not None:
                fc_ex_in.grad.zero_()
            if a_in.grad is not None:
                a_in.grad.zero_()
            
            # Predicted controls from the neural network.
            fc_in.requires_grad = True
            fc_ex_in.requires_grad = True
            a_in.requires_grad = True
            
            sources = ks(f0_in, fc_in, onset_frame_indices_in, fc_ex_in, a_in)
            # Dummy cost function
            error = sources.sum()
            
            #print("Do backward pass.")
            error.backward()
        
            # Disable gradient tracking.
            ks.detach()


if __name__ == "__main__":
    unittest.main()
    
    #test = TestCore()
    #test.test_KS_synthetic_input()
    #test.test_KS_differentiability()
    
# PyTorch implementation of DDSP following closely the original code
# https://github.com/magenta/ddsp/blob/master/ddsp/synths.py

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import scipy.signal
import math

from ddsp import processors
from ddsp import core, spectral_ops
#import processors
#import core, spectral_ops

import matplotlib.pyplot as plt

class Harmonic(processors.Processor):
    """Synthesize audio with a bank of harmonic sinusoidal oscillators."""

    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 scale_fn_amplitudes=core.exp_sigmoid,
                 scale_fn_distribution=core.exp_sigmoid,
                 normalize_below_nyquist=True):
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.scale_fn_amplitudes = scale_fn_amplitudes
        self.scale_fn_distribution = scale_fn_distribution
        self.normalize_below_nyquist = normalize_below_nyquist

    def get_controls(self,
                     amplitudes,
                     harmonic_distribution,
                     f0_hz):
        """Convert network output tensors into a dictionary of synthesizer controls.
        Args:
          amplitudes: 3-D Tensor of synthesizer controls, of shape
            [batch, time, 1].
          harmonic_distribution: 3-D Tensor of synthesizer controls, of shape
            [batch, time, n_harmonics].
          f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].
        Returns:
          controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the amplitudes.
        if self.scale_fn_amplitudes is not None:
            amplitudes = self.scale_fn_amplitudes(amplitudes)
        if self.scale_fn_distribution is not None:
            harmonic_distribution = self.scale_fn_distribution(harmonic_distribution)

        # Bandlimit the harmonic distribution.
        if self.normalize_below_nyquist:
            n_harmonics = int(harmonic_distribution.shape[-1])
            harmonic_frequencies = core.get_harmonic_frequencies(f0_hz,
                                                                 n_harmonics)
            harmonic_distribution = core.remove_above_nyquist(harmonic_frequencies,
                                                              harmonic_distribution,
                                                              self.sample_rate)

        # Normalize
        harmonic_distribution /= torch.sum(harmonic_distribution,
                                           dim=-1,
                                           keepdim=True)

        return {'amplitudes': amplitudes,
                'harmonic_distribution': harmonic_distribution,
                'f0_hz': f0_hz}

    def get_signal(self, amplitudes, harmonic_distribution, f0_hz):
        """Synthesize audio with additive synthesizer from controls.
        Args:
          amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
            float32 that is strictly positive.
          harmonic_distribution: Tensor of shape [batch, n_frames, n_harmonics].
            Expects float32 that is strictly positive and normalized in the last
            dimension.
          f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
            n_frames, 1].
        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        signal = core.harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            harmonic_distribution=harmonic_distribution,
            n_samples=self.n_samples,
            sample_rate=self.sample_rate)
        return signal


class Sinusoidal(processors.Processor):
    """Synthesize audio with a bank of arbitrary sinusoidal oscillators."""

    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 amp_scale_fn=core.exp_sigmoid,
                 amp_resample_method='window',
                 freq_scale_fn=core.frequencies_softmax):

        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.amp_scale_fn = amp_scale_fn
        self.amp_resample_method = amp_resample_method
        self.freq_scale_fn = freq_scale_fn

    def get_controls(self, amplitudes, frequencies):
        """Convert network output tensors into a dictionary of synthesizer controls.
        Args:
          amplitudes: 3-D Tensor of synthesizer controls, of shape
            [batch, time, n_sinusoids].
          frequencies: 3-D Tensor of synthesizer controls, of shape
            [batch, time, n_sinusoids]. Expects strictly positive in Hertz.
        Returns:
          controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the inputs.
        if self.amp_scale_fn is not None:
            amplitudes = self.amp_scale_fn(amplitudes)

        if self.freq_scale_fn is not None:
            frequencies = self.freq_scale_fn(frequencies)
            amplitudes = core.remove_above_nyquist(frequencies,
                                                   amplitudes,
                                                   self.sample_rate)

        return {'amplitudes': amplitudes,
                'frequencies': frequencies}

    def get_signal(self, amplitudes, frequencies):
        """Synthesize audio with sinusoidal synthesizer from controls.
        Args:
          amplitudes: Amplitude tensor of shape [batch, n_frames, n_sinusoids].
            Expects float32 that is strictly positive.
          frequencies: Tensor of shape [batch, n_frames, n_sinusoids].
            Expects float32 in Hertz that is strictly positive.
        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        # Create sample-wise envelopes.
        amplitude_envelopes = core.resample(amplitudes, self.n_samples,
                                            method=self.amp_resample_method)
        frequency_envelopes = core.resample(frequencies, self.n_samples)

        signal = core.oscillator_bank(frequency_envelopes=frequency_envelopes,
                                      amplitude_envelopes=amplitude_envelopes,
                                      sample_rate=self.sample_rate)
        return signal


class FilteredNoise(processors.Processor):
    """Synthesize audio by filtering white noise."""

    def __init__(self,
                 n_samples=64000,
                 window_size=257,
                 scale_fn=core.exp_sigmoid,
                 initial_bias=-5.0):
        super().__init__()
        self.n_samples = n_samples
        self.window_size = window_size
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias

    def get_controls(self, magnitudes):
        """Convert network outputs into a dictionary of synthesizer controls.
        Args:
          magnitudes: 3-D Tensor of synthesizer parameters, of shape [batch, time,
            n_filter_banks].
        Returns:
          controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the magnitudes.
        if self.scale_fn is not None:
            magnitudes = self.scale_fn(magnitudes + self.initial_bias)

        return {'magnitudes': magnitudes}

    def get_signal(self, magnitudes):
        """Synthesize audio with filtered white noise.
        Args:
          magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
            Expects float32 that is strictly positive.
        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        batch_size = int(magnitudes.shape[0])
        signal = torch.rand(size=[batch_size, self.n_samples], device=magnitudes.device) * 2 - 1  # uniform in [-1, 1)
        # TODO shouldn't the magnitude frames be synchronized with the input representation frames (e.g. STFT frames)?
        return core.frequency_filter(signal,
                                     magnitudes,
                                     window_size=self.window_size)



class GainNoise(processors.Processor):
    """Synthesize audio as white noise with time-varying gain."""

    def __init__(self,
                 n_samples=64000,
                 scale_fn=functools.partial(core.exp_sigmoid, max_value=1.0),
                 initial_bias=0.0):
        super().__init__()
        self.n_samples = n_samples
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias

    def get_controls(self, gain):
        """Convert network outputs into a dictionary of synthesizer controls.
        Args:
          gain: 3-D Tensor of synthesizer parameters, of shape [batch, time, 1].
        Returns:
          controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the gain.
        if self.scale_fn is not None:
            gain = self.scale_fn(gain + self.initial_bias)

        return {'gain': gain}

    def get_signal(self, gain):
        """Synthesize audio with filtered white noise.
        Args:
          gain: gain tensor of shape [batch, n_frames, 1].
            Expects float32 that is strictly positive.
        Returns:
          signal: A tensor of noise with time-varying gain [batch, n_samples].
        """
        batch_size, n_frames, _ = gain.shape
        signal = torch.rand(size=[batch_size, self.n_samples], device=gain.device) * 2 - 1  # uniform in [-1, 1)

        # Cut audio into frames.
        frame_size = int(np.ceil(self.n_samples / n_frames))
        audio_frames = torch.split(signal, frame_size, dim=1)  # tuple of tensors, last one might be shorter
        audio_frames = list(audio_frames)

        # zero-pad last frame if necessary
        last_frame_size = audio_frames[-1].shape[-1]
        if last_frame_size < frame_size:
            pad_length = frame_size - last_frame_size
            last_frame_padded = torch.nn.functional.pad(audio_frames[-1], pad=(0, pad_length))
            audio_frames = audio_frames[:-1] + [last_frame_padded]
        else: pad_length = 0

        audio_frames = torch.stack(audio_frames, dim=1)  # [batch_size, n_frames, frame_size]

        # apply gain frame-wise
        audio_frames = audio_frames * gain

        signal = audio_frames.reshape((batch_size, -1))
        if pad_length > 0: signal = signal[:, :-pad_length]

        return signal


class VoiceUnvoicedNoise(processors.Processor):
    """ Synthesize white noise with time-varying gain.
        The noise is high-pass filtered for voiced frames
        and not filtered for unvoiced frames. The frame length
        should be the same as in the STFT applied to the input signal
        to be re-synthesised. A 50 % overlap is applied by default.
         """

    def __init__(self,
                 n_samples=64000,
                 frame_length=512,
                 scale_fn=functools.partial(core.exp_sigmoid, max_value=1.0),
                 initial_bias=0.0,
                 hp_cutoff=500):

        super().__init__()
        self.n_samples = n_samples
        self.frame_length = frame_length  # frame length for which gain and voiced_unvoiced are estimated
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias
        self.hp_cutoff = hp_cutoff  # cutoff frequency for the high pass filter

        # compute high-pass filter coefficients
        b, a = scipy.signal.cheby2(N=8, rs=60, Wn=self.hp_cutoff, btype='highpass', fs=16000)
        self.a = a
        self.b = b

    def get_controls(self, gain, voiced_unvoiced):
        """Convert network outputs into a dictionary of synthesizer controls.
        Args:
          gain: 3-D Tensor of synthesizer parameters, of shape [batch_size, n_frames, 1].
          voiced_unvoiced: torch.Tensor [batch_size, n_frames, 1]. Should be 1 if a frame is
            voiced and 0 if a frame is unvoiced.
        Returns:
          controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the gain.
        if self.scale_fn is not None:
            gain = self.scale_fn(gain + self.initial_bias)

        return {'noise_gain': gain, 'voiced_unvoiced': voiced_unvoiced}

    def get_signal(self, noise_gain, voiced_unvoiced):
        """Synthesize audio with filtered white noise.
        Args:
            noise_gain: gain tensor of shape [batch_size, n_frames, 1].
                Expects float32 that is strictly positive.
            voiced_unvoiced: torch.Tensor [batch_size, n_frames, 1]. Should be 1 if a frame is
            voiced and 0 if a frame is unvoiced.
        Returns:
            signal: A tensor with noise signal with time-varying gain
                and high-pass filter applied for voiced frames [batch, n_samples].
        """
        batch_size, n_gain_frames, _ = noise_gain.shape
        device = noise_gain.device
        white_noise = torch.rand(size=[batch_size, self.n_samples], device=device) * 2 - 1  # uniform in [-1, 1[

        # Pad audio so that frames can slide like in tf.stft(pad=True) and last frame is complete
        hop_size = self.frame_length// 2  # 50 % overlap is assumed by default for window COLA
        white_noise = core.pad_for_stft(white_noise, self.frame_length, hop_size)
        padded_length = white_noise.shape[1]

        # Cut audio into frames.
        white_noise = white_noise[:, None, None, :]  # add a channel dim and a spatial dim for torch.unfold (requires 4D input)
        white_noise_frames = torch.nn.functional.unfold(white_noise, kernel_size=(1, self.frame_length), stride=(1, hop_size))  # [batch_size, frame_size, n_frames]
        white_noise_frames = white_noise_frames.transpose(1, 2)  # [batch_size, n_frames, frame_size]

        batch_size, n_audio_frames, frame_size = white_noise_frames.shape

        # Check that number of frames match.
        if n_audio_frames != n_gain_frames:
            raise ValueError(
                'Number of Audio frames ({}) and gain frames ({}) do not '
                'match.'.format(n_audio_frames, n_gain_frames))

        # generate high-pass filtered white noise
        hp_noise_frames = np.random.rand(batch_size, n_audio_frames, frame_size) * 2 - 1  # uniform in [-1, 1[
        hp_noise_frames = scipy.signal.lfilter(self.b, self.a, hp_noise_frames, axis=-1)
        hp_noise_frames = torch.tensor(hp_noise_frames, dtype=torch.float32, device=device)

        #  use high-pass filtered noise in voiced frames, white noise in unvoiced frames
        noise_frames = torch.where(voiced_unvoiced > 0, hp_noise_frames, white_noise_frames)

        #  apply gain
        noise_frames = noise_frames * noise_gain

        # window
        hann_window = torch.hann_window(self.frame_length, periodic=True, device=device)[None, None, :]
        noise_frames[:, 1:, :] = noise_frames[:, 1:, :] * hann_window
        # first half of first frame is not windowed because nothing is added here in overlap-add
        noise_frames[:, 0, hop_size:] = noise_frames[:, 0, hop_size:] * hann_window[:, :, hop_size:]

        # overlap add back together
        noise_frames = torch.transpose(noise_frames, 1, 2)  # [batch_size, frame_size, n_frames]
        noise = torch.nn.functional.fold(noise_frames,
                                             output_size=(1, padded_length),
                                             kernel_size=(1, self.frame_length),
                                             stride=(1, hop_size))

        noise = noise[:, 0, 0, :self.n_samples]
        return noise


class VoiceUnvoicedNoise2(processors.Processor):
    """ Synthesize white noise with time-varying gain
        and estimated time-invariant noise shape for voiced frames.
        Unvoiced frames are not filtered. The frame length
        should be the same as in the STFT applied to the input signal
        to be re-synthesised. A 50 % overlap is applied by default.
         """

    def __init__(self,
                 n_samples=64000,
                 frame_length=512,
                 initial_bias=0.0,
                 n_magnitudes=40):

        super().__init__()
        self.n_samples = n_samples
        self.frame_length = frame_length  # frame length for which gain and voiced_unvoiced are estimated
        self.initial_bias = initial_bias
        self.n_magnitudes = n_magnitudes  # number of estimated noise magnitude (evenly spaced from 0 to fs/2)

    def get_controls(self, gain, magnitudes, voiced_unvoiced):
        """Convert network outputs into a dictionary of synthesizer controls.
        Args:
          gain: 3-D Tensor of synthesizer parameters, of shape [batch_size, n_frames, 1].
          magnitudes: torch.Tensor [batch_size, 1, n_magnitudes] a set of magnitudes
              specifying a time invariant noise shape for voiced sounds.
          voiced_unvoiced: torch.Tensor [batch_size, n_frames, 1]. Should be 1 if a frame is
            voiced and 0 if a frame is unvoiced.
        Returns:
          controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the gain.
        gain = core.exp_sigmoid(gain + self.initial_bias, max_value=1.)

        magnitudes = core.exp_sigmoid(magnitudes * 2.5)

        return {'noise_gain': gain, 'voiced_noise_magnitudes': magnitudes,
                'voiced_unvoiced': voiced_unvoiced}

    def get_signal(self, noise_gain, voiced_noise_magnitudes, voiced_unvoiced):
        """Synthesize audio with filtered white noise.
        Args:
            noise_gain: gain tensor of shape [batch_size, n_frames, 1].
                Expects float32 that is strictly positive.
            voiced_noise_magnitudes: torch.Tensor [batch_size, 1, n_magnitudes] a set of magnitudes
                specifying a time invariant noise shape for voiced sounds.
            voiced_unvoiced: torch.Tensor [batch_size, n_frames, 1]. Should be 1 if a frame is
                voiced and 0 if a frame is unvoiced.
        Returns:
            signal: A tensor with noise signal with time-varying gain
                and FIR filter with specified magnitudes applied for
                voiced frames [batch, n_samples].
        """
        batch_size, n_gain_frames, _ = noise_gain.shape
        device = noise_gain.device
        white_noise = torch.rand(size=[batch_size, self.n_samples], device=device) * 2 - 1  # uniform in [-1, 1[

        # Pad audio so that frames can slide like in tf.stft(pad=True) and last frame is complete
        hop_size = self.frame_length// 2  # 50 % overlap is assumed by default for window COLA
        white_noise = core.pad_for_stft(white_noise, self.frame_length, hop_size)
        padded_length = white_noise.shape[1]

        # Cut audio into frames.
        white_noise = white_noise[:, None, None, :]  # add a channel dim and a spatial dim for torch.unfold (requires 4D input)
        white_noise_frames = torch.nn.functional.unfold(white_noise, kernel_size=(1, self.frame_length), stride=(1, hop_size))  # [batch_size, frame_size, n_frames]
        white_noise_frames = white_noise_frames.transpose(1, 2)  # [batch_size, n_frames, frame_size]

        batch_size, n_audio_frames, frame_size = white_noise_frames.shape

        # Check that number of frames match.
        if n_audio_frames != n_gain_frames:
            raise ValueError(
                'Number of Audio frames ({}) and gain frames ({}) do not '
                'match.'.format(n_audio_frames, n_gain_frames))

        # generate white noise filtered with specified FIR filter
        voiced_noise = torch.rand(size=white_noise.shape, device=device) * 2 - 1  # uniform in [-1, 1[
        voiced_noise = core.frequency_filter(voiced_noise[:, 0, 0, :], voiced_noise_magnitudes)[:, None, None, :]
        voiced_noise_frames = torch.nn.functional.unfold(voiced_noise, kernel_size=(1, self.frame_length), stride=(1, hop_size))  # [batch_size, frame_size, n_frames]
        voiced_noise_frames = voiced_noise_frames.transpose(1, 2)  # [batch_size, n_frames, frame_size]

        #  use filtered noise in voiced frames, white noise in unvoiced frames
        noise_frames = torch.where(voiced_unvoiced > 0, voiced_noise_frames, white_noise_frames)

        #  apply gain
        noise_frames = noise_frames * noise_gain

        # window
        hann_window = torch.hann_window(self.frame_length, periodic=True, device=device)[None, None, :]
        noise_frames[:, 1:, :] = noise_frames[:, 1:, :] * hann_window
        # first half of first frame is not windowed because nothing is added here in overlap-add
        noise_frames[:, 0, hop_size:] = noise_frames[:, 0, hop_size:] * hann_window[:, :, hop_size:]

        # overlap add back together
        noise_frames = torch.transpose(noise_frames, 1, 2)  # [batch_size, frame_size, n_frames]
        noise = torch.nn.functional.fold(noise_frames,
                                         output_size=(1, padded_length),
                                         kernel_size=(1, self.frame_length),
                                         stride=(1, hop_size))

        noise = noise[:, 0, 0, :self.n_samples]
        return noise


class SourceFilterSynth(processors.Processor):

    """Synthesize audio with a source filter model
       H(z)(I(z)C(z) + N(z)D(z))
       H: all-pole filter defined by reflection coefficients
       I: sum of harmonic sinusoids
       C: harmonic distribution filter
       N: white noise
       D: time-varying gain or FIR filter
    """

    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 n_harmonics=101,
                 audio_frame_size=512,
                 filtered_noise=False,
                 parallel_iir=False,
                 harmonic_distribution_activation=core.exp_sigmoid):

        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.audio_frame_size = audio_frame_size
        self.parallel_iir = parallel_iir

        self.harmonic_synth = Harmonic(n_samples, sample_rate, scale_fn_distribution=harmonic_distribution_activation)
        if filtered_noise: self.noise_synth = FilteredNoise(n_samples)
        else: self.noise_synth = GainNoise(n_samples)

    def get_controls(self,
                     harmonic_amplitudes,
                     harmonic_distribution,
                     f0_hz,
                     noise_control,
                     reflection_coeff):
        """Convert network output tensors into a dictionary of synthesizer controls
           by applying control-specific activation functions.

        Args:
              harmonic_amplitudes: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1].
              harmonic_distribution: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, n_harmonics].
              f0_hz: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1]. Expects strictly positive in Hertz.
              noise_control: 3-D Tensor of shape [batch_size, n_frames, n_controls]
                where n_controls = 1 if the gain noise synth is used. If the
                filtered noise synth is used, n_controls is the number of
                frequency bands of the filter.
              reflection_coeff: 3-D Tensor of shape [batch_size, n_frames, n_coeff]
        Returns:
            controls: Dictionary of tensors of synthesizer controls.
        """

        # harmonic model
        harmonics_control_dict = self.harmonic_synth.get_controls(harmonic_amplitudes, harmonic_distribution, f0_hz)

        # noise model
        noise_control_dict = self.noise_synth.get_controls(noise_control)
        noise_control = list(noise_control_dict.values())[0]

        # vocal tract filter
        reflection_coeff = torch.tanh(reflection_coeff)

        return {'harmonic_amplitudes': harmonics_control_dict['amplitudes'],
                'harmonic_distribution': harmonics_control_dict['harmonic_distribution'],
                'f0_hz': harmonics_control_dict['f0_hz'],
                'noise_control': noise_control,
                'reflection_coeff': reflection_coeff}

    def get_signal(self,
                   harmonic_amplitudes,
                   harmonic_distribution,
                   f0_hz,
                   noise_control,
                   reflection_coeff):

        """Synthesize audio with sinusoidal synthesizer from controls.
        Args:
            harmonic_amplitudes: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1].
            harmonic_distribution: 3-D Tensor of shape [batch_size, n_frames, n_harmonics]
            f0_hz: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1]. Expects strictly positive in Hertz.
            noise_control: 3-D Tensor of shape [batch_size, n_frames, n_controls]
            reflection_coeff: 3-D Tensor of shape [batch_size, n_frames, n_coeff]
        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples].
        """

        harmonics = self.harmonic_synth.get_signal(harmonic_amplitudes, harmonic_distribution, f0_hz)
        noise = self.noise_synth.get_signal(noise_control)
        source = harmonics + noise
        signal = core.filter_with_all_pole(source, reflection_coeff, self.audio_frame_size, parallel=self.parallel_iir)
        return signal


class SourceFilterSynthLSF(processors.Processor):

    """Synthesize audio with a source filter model
       H(z)(I(z)C(z) + N(z)D(z))
       H: all-pole filter defined by Line Spectral Frequencies (LSFs)
       I: sum of harmonic sinusoids
       C: harmonic distribution filter
       N: white noise
       D: time-varying gain or FIR filter
    """

    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 n_harmonics=101,
                 audio_frame_size=512,
                 filtered_noise=False,
                 parallel_iir=False,
                 harmonic_distribution_activation=core.exp_sigmoid):

        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.audio_frame_size = audio_frame_size
        self.parallel_iir = parallel_iir

        self.harmonic_synth = Harmonic(n_samples, sample_rate, scale_fn_distribution=harmonic_distribution_activation)
        if filtered_noise: self.noise_synth = FilteredNoise(n_samples)
        else: self.noise_synth = GainNoise(n_samples)

    def get_controls(self,
                     harmonic_amplitudes,
                     harmonic_distribution,
                     f0_hz,
                     noise_control,
                     line_spectral_frequencies):
        """Convert network output tensors into a dictionary of synthesizer controls
           by applying control-specific activation functions.

        Args:
              harmonic_amplitudes: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1].
              harmonic_distribution: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, n_harmonics].
              f0_hz: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1]. Expects strictly positive in Hertz.
              noise_control: 3-D Tensor of shape [batch_size, n_frames, n_controls]
                where n_controls = 1 if the gain noise synth is used. If the
                filtered noise synth is used, n_controls is the number of
                frequency bands of the filter.
              line_spectral_frequencies: 3-D Tensor of shape [batch_size, n_frames, n_lsf + 1]
        Returns:
            controls: Dictionary of tensors of synthesizer controls.
        """

        # harmonic model
        harmonics_control_dict = self.harmonic_synth.get_controls(harmonic_amplitudes, harmonic_distribution, f0_hz)

        # noise model
        noise_control_dict = self.noise_synth.get_controls(noise_control)
        noise_control = list(noise_control_dict.values())[0]

        # vocal tract filter
        line_spectral_frequencies = core.lsf_activation(line_spectral_frequencies)

        return {'harmonic_amplitudes': harmonics_control_dict['amplitudes'],
                'harmonic_distribution': harmonics_control_dict['harmonic_distribution'],
                'f0_hz': harmonics_control_dict['f0_hz'],
                'noise_control': noise_control,
                'line_specral_frequencies': line_spectral_frequencies}

    def get_signal(self,
                   harmonic_amplitudes,
                   harmonic_distribution,
                   f0_hz,
                   noise_control,
                   line_specral_frequencies):

        """Synthesize audio with sinusoidal synthesizer from controls.
        Args:
            harmonic_amplitudes: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1].
            harmonic_distribution: 3-D Tensor of shape [batch_size, n_frames, n_harmonics]
            f0_hz: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1]. Expects strictly positive in Hertz.
            noise_control: 3-D Tensor of shape [batch_size, n_frames, n_controls]
            line_specral_frequencies: 3-D Tensor of shape [batch_size, n_frames, n_lsf]
        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples].
        """

        harmonics = self.harmonic_synth.get_signal(harmonic_amplitudes, harmonic_distribution, f0_hz)
        noise = self.noise_synth.get_signal(noise_control)
        source = harmonics + noise
        filter_coeff = core.lsf_to_filter_coeff(line_specral_frequencies)
        signal = core.apply_all_pole_filter(source, filter_coeff, self.audio_frame_size, parallel=self.parallel_iir)
        return signal



class SourceFilterSynth2(processors.Processor):

    """Synthesize audio with a source filter model
       H(z)(I(z)C(z) + N(z)D(z))  --> voiced
       H(z)N(z)                   --> unvoiced
       H: all-pole filter defined by Line Spectral Frequencies (LSFs)
       I: sum of harmonic sinusoids
       C: FIR filter for glottal source roll off (x dB/octave)
       N: white noise
       D: high-pass filter
       The voiced/unvoiced decision is taken through the f0 estimate
    """

    def __init__(self,
                 n_samples=64000,
                 sample_rate=16000,
                 n_harmonics=101,
                 audio_frame_size=512,
                 hp_cutoff=500,
                 f_ref=500,
                 estimate_voiced_noise_mag=True):

        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.audio_frame_size = audio_frame_size
        self.f_ref = f_ref  # reference frequency for harmonic roll off
        self.estimate_voiced_noise_mag = estimate_voiced_noise_mag

        self.harmonic_synth = Harmonic(n_samples, sample_rate, scale_fn_distribution=core.exp_sigmoid)

        if estimate_voiced_noise_mag: self.noise_synth = VoiceUnvoicedNoise2(n_samples, audio_frame_size, n_magnitudes=40)
        else: self.noise_synth = VoiceUnvoicedNoise(n_samples, audio_frame_size, hp_cutoff=hp_cutoff)

    def get_controls(self,
                     harmonic_amplitudes,
                     harmonics_roll_off,
                     f0_hz,
                     noise_gain,
                     voiced_unvoiced,
                     line_spectral_frequencies,
                     *voiced_noise_magnitudes):

        """Convert network output tensors into a dictionary of synthesizer controls
           by applying control-specific activation functions.

        Args:
              harmonic_amplitudes: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1].
              harmonics_roll_off: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1]. roll off in dB/octave for harmonics filter
                    specified as positive number
              f0_hz: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1]. Expects strictly positive in Hertz.
              noise_gain: 3-D Tensor of shape [batch_size, n_frames, 1]
              voiced_unvoiced: 3-D Tensor of shape [batch_size, n_frames, 1],
                    should be equal to 1 if the frame is voiced and 0 if the
                    frame is unvoiced.
              line_spectral_frequencies: 3-D Tensor of shape [batch_size, n_frames, n_lsf + 1]
              voiced_noise_magnitudes: (optional) torch.Tensor [batch_size, 1, n_magnitudes]
        Returns:
            controls: Dictionary of tensors of synthesizer controls.
        """

        # harmonic model
        harmonic_distribution = torch.ones((1, 1, self.n_harmonics), dtype=torch.float32, device=harmonic_amplitudes.device)
        harmonics_control_dict = self.harmonic_synth.get_controls(harmonic_amplitudes, harmonic_distribution, f0_hz)

        # noise model
        if self.estimate_voiced_noise_mag: noise_control_dict = \
              self.noise_synth.get_controls(noise_gain, voiced_noise_magnitudes[0], voiced_unvoiced)
        else: noise_control_dict = self.noise_synth.get_controls(noise_gain, voiced_unvoiced)

        # vocal tract filter
        line_spectral_frequencies = core.lsf_activation(line_spectral_frequencies)

        return {'harmonic_amplitudes': harmonics_control_dict['amplitudes'],
                'harmonic_distribution': harmonics_control_dict['harmonic_distribution'],
                'f0_hz': harmonics_control_dict['f0_hz'],
                'harmonics_roll_off': harmonics_roll_off,
                'line_spectral_frequencies': line_spectral_frequencies,
                **noise_control_dict,
                }

    def get_signal(self,
                   harmonic_amplitudes,
                   harmonic_distribution,
                   f0_hz,
                   harmonics_roll_off,
                   line_spectral_frequencies,
                   noise_gain,
                   voiced_unvoiced,
                   **kwargs
                   ):

        """Synthesize audio with sinusoidal synthesizer from controls.
        Args:
            harmonic_amplitudes: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1].
            harmonic_distribution: 3-D Tensor of shape [batch_size, n_frames, n_harmonics]
            f0_hz: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1]. Expects strictly positive in Hertz.
            harmonics_roll_off: 3-D Tensor of synthesizer controls, of shape
                    [batch, n_frames, 1]. roll off in dB/octave for harmonics filter
                    specified as positive number
            noise_gain: 3-D Tensor of shape [batch_size, n_frames, 1]
            voiced_unvoiced: 3-D Tensor of shape [batch_size, n_frames, 1],
                    should be equal to 1 if the frame is voiced and 0 if the
                    frame is unvoiced.
            line_spectral_frequencies: 3-D Tensor of shape [batch_size, n_frames, n_lsf]
        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples].
        """

        harmonics = self.harmonic_synth.get_signal(harmonic_amplitudes, harmonic_distribution, f0_hz)


        filter_mag = core.slope_frequency_response(harmonics_roll_off, n_freqs=65, f_ref=self.f_ref)
        harmonics = core.frequency_filter(harmonics, filter_mag)

        if self.estimate_voiced_noise_mag: noise = self.noise_synth.get_signal(noise_gain,
                                                                               kwargs['voiced_noise_magnitudes'],
                                                                               voiced_unvoiced)
        else: noise = self.noise_synth.get_signal(noise_gain, voiced_unvoiced)

        source = harmonics + noise

        filter_coeff = core.lsf_to_filter_coeff(line_spectral_frequencies)
        signal = core.apply_all_pole_filter(source, filter_coeff, self.audio_frame_size, parallel=True)
        return signal


class KarplusStrong(processors.Processor):
    """
    A simple Karplus-Strong string model.
    
    Args:
        batch_size: int, batch size of the model
        n_samples: int, number of time samples per training example
        sample_rate: int, sample rate in Hertz
        audio_frame_size: int, number of time samples in one (STFT) audio frame
        n_strings: int, number of strings
        min_freq: minimum frequency that can be synthesized
        excitation_length: Length of the excitation signal in seconds
        excitation_amplitude_scale: Maximum value of the excitation amplitude factor
    """
    def __init__(self,
                 batch_size=4,
                 n_samples=64000,
                 sample_rate=16000,
                 audio_frame_size=256,
                 n_strings=6,
                 min_freq=20,
                 excitation_length=0.005,
                 excitation_amplitude_scale=10):
        assert n_samples % audio_frame_size == 0.0, \
            f"The n_samples must be a multiple of audio_frame_size!\nBut n_samples is {n_samples} and audio_frame_size is {audio_frame_size}."
        
        super().__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.audio_frame_size = audio_frame_size
        self.n_strings = n_strings
        self.min_freq = min_freq
        self.excitation_amplitude_scale = excitation_amplitude_scale
        
        # Delay line
        self.max_delay = 1 / min_freq
        self.dl = core.DelayLine(batch_size=batch_size,
                                 n_delays=n_strings,
                                 length=self.max_delay,
                                 sr=sample_rate)
        
        # Feedback lowpass
        self.lp = core.SimpleLowpass(batch_size=batch_size,
                                     n_filters=n_strings,
                                     sr=sample_rate)
        # DC blocking feedback highpass
        self.hp = core.SimpleHighpass(batch_size=batch_size,
                                      n_filters=n_strings,
                                      sr=sample_rate)
        hp_fc = 20
        self.hp.set_fc(hp_fc * torch.ones((batch_size, n_strings)))
        
        # Excitation
        self.n_excitation_samples = math.ceil(excitation_length * sample_rate)
        # Use just white noise as excitation signal by now...
        self.excitation = torch.rand(self.n_excitation_samples) * 2 - 1
        
        # Excitation filter
        self.lp_ex = core.SimpleLowpass(batch_size=batch_size,
                                        n_filters=n_strings,
                                        sr=sample_rate)
        
        self.excitation_block = torch.zeros((self.batch_size, self.n_strings,
                                        self.n_samples + self.n_excitation_samples))
        # An excitation signal may reach into the next train example.
        self.last_excitation_overhead = torch.zeros(batch_size,
                                                    n_strings,
                                                    self.n_excitation_samples)
        
        self.last_valid_f0 = torch.ones((batch_size, n_strings)) * min_freq
    
    def detach(self):
        """
        Detach the internal state of the synth from the current graph.
        """
        # DDSP objects
        self.dl.detach()
        self.lp.detach()
        self.hp.detach()
        self.lp_ex.detach()
        
        # Pytorch tensors
        self.excitation_block = self.excitation_block.detach()
        self.last_excitation_overhead = self.last_excitation_overhead.detach()
        
    
    def clear_state(self):
        """
        Set the internal state of the synth to zero.
        """
        for element in self.internal_state:
            if type(element) == torch.Tensor or type(element) == torch.tensor:
                element = torch.zeros(*element.shape)
            else:
                element.clear_state()
    

    def get_controls(self, f0_hz, fc, onset_frame_indices, fc_ex, a, g):
        """
        Convert network output tensors into a dictionary of synthesizer controls.
        Args:
            f0_hz: Fundamental frequencies in Hertz,
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
            fc:    Loop filter cutoff frequency scaled to [0, 1],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
            onset_frame_indices: Note onset frame indices to trigger excitation
                signals. One index is [batch, string, onset_frame],
                torch.tensor of shape [n_onset_indices, 3]
            fc_ex: Excitation filter cutoff frequency scaled to [0, 1],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
            a: Excitation amplitude factor scaled to [0, 1],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
            g: Feedback gain factor between [0, 1],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
        """
        n_frames = f0_hz.shape[2]
        
        # If note is off, stay at last valid fundamental frequency.
        for batch in range(self.batch_size):
            for string in range(self.n_strings):
                for frame in range(n_frames):
                    if f0_hz[batch, string, frame] == 0:
                        f0_hz[batch, string, frame] = self.last_valid_f0[batch, string]
                    else:
                        self.last_valid_f0[batch, string] = f0_hz[batch, string, frame]
        
        # Scale parameters
        fc = fc * self.sample_rate / 2
        fc_ex = fc_ex * self.sample_rate / 2
        a = a * self.excitation_amplitude_scale
        
        # Fundamental periods
        t0 = core.safe_divide(1, f0_hz)
        # Limit fundamental period to maximum delay
        t0 [t0 > self.max_delay] = self.max_delay
        
        return {"t0": t0,
                'fc': fc,
                'onset_frame_indices': onset_frame_indices,
                'fc_ex': fc_ex,
                'a': a,
                'g': g}
    

    def get_signal(self, t0, fc, onset_frame_indices, fc_ex, a, g, **kwargs):
        """
        Synthesize one train example from the given arguments.
        
        Args:
            t0: Fundamental periods of the played notes in seconds,
                 torch.Tensor of shape [batch_size, n_strings, n_frames]
            fc: Loop filter cutoff frequency in Hertz,
                 torch.Tensor of shape [batch_size, n_strings, n_frames]
            onset_frame_indices: Note onset frame indices to trigger excitation
                 signals. One index is [batch, string, onset_frame],
                 torch.tensor of shape [n_onset_indices, 3]
            fc_ex: Excitation filter cutoff frequency in Hertz,
                 torch.Tensor of shape [batch_size, n_strings, n_frames]
            a: Excitation amplitude factor,
                 torch.Tensor of shape [batch_size, n_strings, n_frames]
            g: Feedback gain factor between [0, 1],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
        
        Returns:
            The synthesized example of string sounds from the given parameters,
            torch.tensor of shape [batch_size, n_strings, n_samples]
        """
        # Build excitations for the training example.
        self.excitation_block = torch.zeros((self.batch_size, self.n_strings,
                                        self.n_samples + self.n_excitation_samples))
        for n, onset_index in enumerate(onset_frame_indices):
            batch = onset_index[0].item()
            string = onset_index[1].item()
            # Convert from frame rate to audio rate.
            start = onset_index[2].item() * self.audio_frame_size
            stop = start + self.n_excitation_samples
            self.excitation_block[batch, string, start : stop] = self.excitation
        
        # Store excitation overhead for next example and resize the
        # excitation block to the correct length.
        self.excitation_block[..., :self.n_excitation_samples] += self.last_excitation_overhead
        self.last_excitation_overhead = self.excitation_block[..., -self.n_excitation_samples:]
        self.excitation_block = self.excitation_block[..., : -self.n_excitation_samples]
        
        # Synthesize audio for every frame
        out = torch.zeros_like(self.excitation_block)
        f = torch.zeros((self.batch_size, self.n_strings))
        last_y = torch.zeros((self.batch_size, self.n_strings))
        n_frames = fc.shape[2]
        for frame_idx in range(n_frames):
            # Set parameters to the played notes.
            t0_f = t0[:, :, frame_idx]
            self.dl.set_delay(t0_f)
            fc_f = fc[:, :, frame_idx]
            self.lp.set_fc(fc_f)
            fc_ex_f = fc_ex[:, :, frame_idx]
            self.lp_ex.set_fc(fc_ex_f)
            a_f = a[:, :, frame_idx]
            g_f = g[:, :, frame_idx]
            
            # Synthesize one frame of audio with the (extended) Karplus-Strong
            # model.
            offset = frame_idx * self.audio_frame_size
            for i in range(self.audio_frame_size):
                excitation = a_f * self.lp_ex(self.excitation_block[..., offset + i])
                f = g_f * self.hp(self.dl(last_y))
                out[..., offset + i] = self.lp(excitation + f)
                last_y = out[..., offset + i]
        return out


class KarplusStrongB(processors.Processor):
    """
    Karplus-Strong string model with original filters and extensions.
    This synth is used in the 'B' experiments.
    
    Args:
        batch_size: int, batch size of the model
        n_samples: int, number of time samples per training example
        sample_rate: int, sample rate in Hertz
        audio_frame_size: int, number of time samples in one (STFT) audio frame
        n_strings: int, number of strings
        min_freq: minimum frequency that can be synthesized
        excitation_length: Length of the excitation signal in seconds
        maximum_excitation_amplitude: Maximum amplitude value of the excitation signal
        maximum_feedback_factor: The maximum value of the loop feedback factor
    """
    def __init__(self,
                 batch_size=4,
                 n_samples=64000,
                 sample_rate=16000,
                 audio_frame_size=256,
                 n_strings=6,
                 min_freq=20,
                 excitation_length=0.005,
                 maximum_excitation_amplitude=0.99,
                 maximum_feedback_factor=0.99):
        assert n_samples % audio_frame_size == 0.0, \
            f"The n_samples must be a multiple of audio_frame_size!\nBut n_samples is {n_samples} and audio_frame_size is {audio_frame_size}."
        
        super().__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.audio_frame_size = audio_frame_size
        self.n_strings = n_strings
        self.min_freq = min_freq
        self.maximum_excitation_amplitude = maximum_excitation_amplitude
        self.maximum_feedback_factor = maximum_feedback_factor
        
        # Delay line
        self.max_delay = 1 / min_freq
        self.dl = core.DelayLine(batch_size=batch_size,
                                 n_delays=n_strings,
                                 length=self.max_delay,
                                 sr=sample_rate)
        
        # Loop lowpass filter
        self.Ha = core.HaDecayStretch(batch_size=batch_size, n_filters=n_strings)
        
        # Excitation dynamics filter
        self.Hd = core.OnePole(batch_size=batch_size, n_filters=n_strings, r=0.5)
        
        # DC blocking loop highpass filter with fc = 20Hz to avoid
        # low frequency ringing.
        self.hp = core.SimpleHighpass(batch_size=batch_size,
                                      n_filters=n_strings,
                                      sr=sample_rate)
        hp_fc = 20
        self.hp.set_fc(hp_fc * torch.ones((batch_size, n_strings)))
        
        # White noise burst excitation signal
        self.n_excitation_samples = math.ceil(excitation_length * sample_rate)
        self.excitation = torch.rand(self.n_excitation_samples) * 2 - 1
        
        self.excitation_block = torch.zeros((self.batch_size, self.n_strings,
                                        self.n_samples + self.n_excitation_samples))
        # An excitation signal may reach into the next train example.
        self.last_excitation_overhead = torch.zeros(batch_size,
                                                    n_strings,
                                                    self.n_excitation_samples)
        
        self.last_valid_f0 = torch.ones((batch_size, n_strings)) * min_freq
    
    def detach(self):
        """
        Detach the internal state of the synth from the current graph.
        """
        # DDSP objects
        self.dl.detach()
        self.Ha.detach()
        self.Hd.detach()
        self.hp.detach()
        
        # Pytorch tensors
        self.excitation_block = self.excitation_block.detach()
        self.last_excitation_overhead = self.last_excitation_overhead.detach()

    def get_controls(self, f0_hz, onset_frame_indices, a, s, r):
        """
        Convert network output tensors into a dictionary of synthesizer controls.
        Args:
            f0_hz: Fundamental frequencies in Hertz,
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
            onset_frame_indices: Note onset frame indices to trigger excitation
                   signals. One index is [batch, string, onset_frame],
                   torch.tensor of shape [n_onset_indices, 3]
            a: Excitation amplitude factor scaled to [0, 1],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
            s: Decay stretching factor scaled to [0, 1],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
            r: Excitation dynamics filter coefficient scaled to [0, 1],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
        """
        n_frames = f0_hz.shape[2]
        
        # If note is off, stay at last valid fundamental frequency.
        for batch in range(self.batch_size):
            for string in range(self.n_strings):
                for frame in range(n_frames):
                    if f0_hz[batch, string, frame] == 0:
                        f0_hz[batch, string, frame] = self.last_valid_f0[batch, string]
                    else:
                        self.last_valid_f0[batch, string] = f0_hz[batch, string, frame]
        
        # Fundamental periods
        t0 = core.safe_divide(1, f0_hz)
        # Limit fundamental period to maximum delay
        t0 [t0 > self.max_delay] = self.max_delay
        
        # Scale parameters
        a = self.maximum_excitation_amplitude * a
        s = 0.1 + 0.9 * s
        r = r * 0.95 # for stability
        
        return {"t0": t0,
                "onset_frame_indices": onset_frame_indices,
                "a": a,
                "s": s,
                "r": r}
    

    def get_signal(self, t0, onset_frame_indices, a, s, r, **kwargs):
        """
        Synthesize one train example from the given arguments.
        
        Args:
            t0: Fundamental periods of the played notes in seconds,
                 torch.Tensor of shape [batch_size, n_strings, n_frames]
            onset_frame_indices: Note onset frame indices to trigger excitation
                 signals. One index is [batch, string, onset_frame],
                 torch.tensor of shape [n_onset_indices, 3]
            a: Excitation amplitude factor,
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
            s: Decay stretching factor scaled to [0, 1],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
            r: Excitation dynamics filter coefficient scaled to [0, r_max],
                   torch.Tensor of shape [batch_size, n_strings, n_frames]
        
        Returns:
            The synthesized example of string sounds from the given parameters,
            torch.tensor of shape [batch_size, n_strings, n_samples]
        """
        # Build excitations for the training example.
        self.excitation_block = torch.zeros((self.batch_size, self.n_strings,
                                        self.n_samples + self.n_excitation_samples))
        for n, onset_index in enumerate(onset_frame_indices):
            batch = onset_index[0].item()
            string = onset_index[1].item()
            # Convert from frame rate to audio rate.
            start = onset_index[2].item() * self.audio_frame_size
            stop = start + self.n_excitation_samples
            self.excitation_block[batch, string, start : stop] = self.excitation
        
        # Store excitation overhead for next example and resize the
        # excitation block to the correct length.
        self.excitation_block[..., :self.n_excitation_samples] += self.last_excitation_overhead
        self.last_excitation_overhead = self.excitation_block[..., -self.n_excitation_samples:]
        self.excitation_block = self.excitation_block[..., : -self.n_excitation_samples]
        
        # Synthesize audio for every frame
        y = torch.zeros_like(self.excitation_block)
        f = torch.zeros((self.batch_size, self.n_strings))
        last_y = torch.zeros((self.batch_size, self.n_strings))
        n_frames = t0.shape[2]
        for frame_idx in range(n_frames):
            # Set t0 and predicted parameters
            t0_in = t0[:, :, frame_idx]
            self.dl.set_delay(t0_in)
            a_in = a[:, :, frame_idx]
            s_in = s[:, :, frame_idx]
            self.Ha.set_coeff(s_in)
            r_in = r[:, :, frame_idx]
            self.Hd.set_coeff(r_in)
            
            # Synthesize one frame of audio with the (extended) Karplus-Strong
            # model.
            offset = frame_idx * self.audio_frame_size
            for i in range(self.audio_frame_size):
                x_e = a_in * self.Hd(self.excitation_block[..., offset + i])
                f = self.maximum_feedback_factor * self.hp(self.dl(last_y))
                # Restrict feedback path for stability.
                f = F.hardtanh(f, min_val=-1, max_val=1)
                y[..., offset + i] = self.Ha(x_e + f)
                last_y = y[..., offset + i]
        return y

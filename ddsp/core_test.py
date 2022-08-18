import numpy as np
from scipy import signal
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math
#import spectrum

import unittest

import core, synths


def plot_spec(y, sample_rate, title=""):
    fig, ax = plt.subplots()
    D_highres = librosa.stft(y.numpy(), hop_length=256, n_fft=4096)
    S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
    img = librosa.display.specshow(S_db_hr, sr=sample_rate, hop_length=256,
                                   x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")


class TestCore(unittest.TestCase):

    def test_fft_convolve_is_accurate(self):
        """Tests convolving signals using fast fourier transform (fft).
        Generate random signals and convolve using fft. Compare outputs to the
        implementation in scipy.signal.

        """
        # Create random signals to convolve.
        audio = np.ones([1, 1000]).astype(np.float32)
        impulse_response = np.ones([1, 50]).astype(np.float32)

        output_pt = core.fft_convolve(
            torch.from_numpy(audio), torch.from_numpy(impulse_response), padding='valid', delay_compensation=0)[0]

        output_np = signal.fftconvolve(audio[0], impulse_response[0])

        difference = output_np - output_pt.numpy()
        total_difference = np.abs(difference).mean()
        # print(total_difference)

        threshold = 1e-3
        self.assertLessEqual(total_difference, threshold)


    def test_reflection_to_filter_coeff(self):

        """
        frame_1: k_1 = 0.5, k_2 = 0.1, k_3 = 0.3  (frame_2: k_1 = 0.4, k_2 = 0.1, k_3 = 0.3)

        i = 1:
            a_1^(1) = k_1 = 0.5  (0.4)

        i = 2:
            a_2^(2) = k_2 = 0.1
                j = 1: a_1^(2) = a_1^(1) - k_2*a_1^(1) = 0.5 - 0.1*0.5 = 0.45  (0.4 - 0.1*0.4 = 0.36)

        i = 3:
            a_3^(3) = k_3 = 0.3
                j = 1: a_1^(3) = a_1^(2) - k_3*a_2^(2) = 0.45 - 0.3*0.1  =  0.42  (0.36 - 0.3*0.1 = 0.33)
                j = 2: a_2^(3) = a_2^(2) - k_3*a_1^(2) = 0.1  - 0.3*0.45 = -0.035 (0.1 - 0.3*0.36 = -0.008)

        """

        reflection_coeff = torch.zeros((2, 2, 3))  # [batch_size, n_frames, n_coeff]
        reflection_coeff[:, 0, 0] = 0.5
        reflection_coeff[:, 0, 1] = 0.1
        reflection_coeff[:, 0, 2] = 0.3
        reflection_coeff[:, 1, 0] = 0.4
        reflection_coeff[:, 1, 1] = 0.1
        reflection_coeff[:, 1, 2] = 0.3

        filter_coeff_expected = torch.tensor([[[0.42, -0.035, 0.3], [0.33, -0.008, 0.3]],
                                              [[0.42, -0.035, 0.3], [0.33, -0.008, 0.3]]]).numpy()
        filter_coeff_computed = core.reflection_to_filter_coeff(reflection_coeff).numpy()

        for i in range(3):
            self.assertAlmostEqual(filter_coeff_expected[0, 0, i], filter_coeff_computed[0, 0, i])
            self.assertAlmostEqual(filter_coeff_expected[1, 0, i], filter_coeff_computed[1, 0, i])
            self.assertAlmostEqual(filter_coeff_expected[0, 1, i], filter_coeff_computed[0, 1, i])
            self.assertAlmostEqual(filter_coeff_expected[1, 1, i], filter_coeff_computed[1, 1, i])


    def test_apply_all_pole_filter(self):
        """
        Test if core.apply_all_pole_filter returns the same filtered signal as scipy.signal.lfilter
        when given the same filter coefficients. This tests the case of a time-invariant filter.
        The filter coefficients are copied in 200 blocks to test the overlap add procedure in
        the function.
        """

        # generate a source signal that will be filtered
        harmonic_synth = synths.Harmonic(n_samples=64000, sample_rate=16000)
        amplitudes = torch.ones((1, 250, 1))  # [batch, n_frames, 1]
        harmonic_distribution = torch.ones((1, 250, 15))  # [batch, n_frames, n_harmonics]
        f0 = torch.ones((1, 250, 1)) * torch.linspace(200, 200, 250)[None, :, None]  # [batch, n_frames, 1]
        source_signal = harmonic_synth(amplitudes, harmonic_distribution, f0)  # [batch_size, 64000]

        # generate filter coefficients that will lead to a stable filter
        b, a = signal.iirfilter(N=5, Wn=0.2, btype='lowpass', output='ba')

        # make all-pole filter by setting b_0 = 1 and b_i = 0 for all i
        b = np.zeros_like(a)
        b[0] = 1.

        y_scipy = signal.lfilter(b=b, a=a, x=source_signal.numpy()[0, :])

        a_torch = torch.tensor(a[1:])[None, None, :]  # remove a_0 and add batch and frame dimensions
        a_torch = torch.cat([a_torch] * 200, dim=1)  # make 200 frames with the same filter coefficients

        audio_block_length = int((64000 / 200) * 2)  # 200 blocks with 50 % overlap --> length=640
        y_test = core.apply_all_pole_filter(source_signal, a_torch, audio_block_size=audio_block_length, parallel=True)
        y_test = y_test[0, :64000].numpy()

        difference = y_scipy - y_test
        total_difference = np.abs(difference).mean()

        threshold = 1e-3
        self.assertLessEqual(total_difference, threshold)


    # def test_lsf_to_filter_coeff(self):

    #     lsf = [0.0483, 0.1020, 0.1240, 0.2139, 0.3012, 0.5279, 0.6416, 0.6953, 0.9224,
    #            1.1515, 1.2545, 1.3581, 1.4875, 1.7679, 1.9860, 2.2033, 2.3631, 2.5655,
    #            2.6630, 2.8564]

    #     a_pyspec = spectrum.lsf2poly(lsf)  # numpy-based method
    #     a_torch = core.lsf_to_filter_coeff(torch.tensor(lsf)[None, None, :])  # PyTorch implementation

    #     a_pyspec = a_pyspec[1:]  # remove 0th coefficient a_0 = 1
    #     a_torch = a_torch.numpy()[0, 0, :]

    #     difference = a_pyspec - a_torch
    #     mean_difference = abs(difference).mean()

    #     threshold = 1e-5
    #     self.assertLessEqual(mean_difference, threshold)

    def test_delay_line_delay(self):
        threshold = 0.1
        
        batch_size = 2
        n_sources = 3
        delay_time = 0.5 # sec
        sr = 16000

        delays = torch.ones((batch_size, n_sources)) * delay_time
        
        dl = core.DelayLine(batch_size, n_sources, delay_time, sr)
        dl.set_delay(delays)
        
        dur = 1 # sec
        f = 1000
        t = torch.linspace(0, dur, int(dur * sr))
        x = torch.sin(2 * np.pi * f * t)
        x = x.unsqueeze(0).repeat(n_sources, 1).unsqueeze(0).repeat(batch_size, 1, 1)
        y = torch.zeros(*x.shape)
        sig_len = x.shape[2]
        for i in range(sig_len):
            y[..., i] = dl(x[..., i])
        
        #DBG:
        #plt.figure()
        #for batch in range(batch_size):
        #    for delay_line in range(n_sources):
        #        plt.plot(y[batch, delay_line].numpy(), alpha=0.4)
        
        
        y_target = torch.cat((torch.zeros(batch_size, n_sources, int(delay_time * sr)),
                              x[..., : math.ceil((dur - delay_time) * sr)]), dim=-1)
        difference = y_target.numpy() - y.numpy()
        mean_difference = abs(difference).mean()
        self.assertLessEqual(mean_difference, threshold)
        
        # change delay
        delay_time = 0.123456789
        
        dl.clear_state()
        dl.set_delay(torch.ones((batch_size, n_sources)) * delay_time)
        y = torch.zeros(*x.shape)
        for i in range(sig_len):
            y[..., i] = dl(x[..., i])
        
        y_target = torch.cat((torch.zeros(batch_size, n_sources, int(delay_time * sr)),
                              x[..., : math.ceil((dur - delay_time) * sr)]), dim=-1)
        difference = y_target.numpy() - y.numpy()
        mean_difference = abs(difference).mean()
        self.assertLessEqual(mean_difference, threshold)
    
    def test_delay_line_differentiability(self):
        batch_size = 2
        n_sources = 3
        delay_time = 0.01 # sec
        sr = 16000

        delays = torch.ones((batch_size, n_sources)) * delay_time
        
        dl = core.DelayLine(batch_size, n_sources, delay_time, sr)
        dl.set_delay(delays)
        
        dur = 0.02 # sec
        sig_len = int(dur * sr)
        x = torch.rand((batch_size, n_sources, sig_len))
        
        for i in range(sig_len):
            x_in = x[..., i]
            # Zeroing out the gradient
            if x_in.grad is not None:
                x_in.grad.zero_()
            
            # Set parameter to calculate gradient
            x_in.requires_grad = True
            
            y = dl(x_in)
            
            # Dummy cost function
            error = y.sum()
            error.backward()
        
            # Detach from current graph.
            dl.detach()
    
    def test_simple_lowpass_differentiability(self):
        batch_size = 2
        n_sources = 3
        fc = 2500
        sr = 16000
        
        # Normally we predict fc with a network and pass it to the filter.
        
        fcs = torch.ones((batch_size, n_sources)) * fc
        
        filt = core.SimpleLowpass(batch_size, n_sources, sr)
        
        dur = 0.02 # sec
        sig_len = int(dur * sr)
        x = torch.rand((batch_size, n_sources, sig_len))
        
        for i in range(sig_len):
            x_in = x[..., i]
            # Zeroing out the gradient (this is normally done with the optimizer)
            if x_in.grad is not None:
                x_in.grad.zero_()
            if fcs.grad is not None:
                fcs.grad.zero_()
            # Detach from current graph.
            filt.detach()
            
            # Set parameter to calculate gradient
            x_in.requires_grad = True
            fcs.requires_grad = True
            filt.set_fc(fcs)
            
            y = filt(x_in)

            # Dummy cost function
            error = y.sum()
            error.backward()

    
    def test_simple_highpass_differentiability(self):
        batch_size = 2
        n_sources = 3
        fc = 2500
        sr = 16000
        
        # Normally we predict fc with a network and pass it to the filter.
        
        fcs = torch.ones((batch_size, n_sources)) * fc
        
        filt = core.SimpleHighpass(batch_size, n_sources, sr)
        
        dur = 0.02 # sec
        sig_len = int(dur * sr)
        x = torch.rand((batch_size, n_sources, sig_len))
        
        for i in range(sig_len):
            x_in = x[..., i]
            # Zeroing out the gradient (this is normally done with the optimizer)
            if x_in.grad is not None:
                x_in.grad.zero_()
            if fcs.grad is not None:
                fcs.grad.zero_()
            # Detach from current graph.
            filt.detach()
            
            # Set parameter to calculate gradient
            x_in.requires_grad = True
            fcs.requires_grad = True
            filt.set_fc(fcs)
            
            y = filt(x_in)

            # Dummy cost function
            error = y.sum()
            error.backward()

    def test_sinc(self):
        x = torch.cat((torch.ones(1), torch.zeros(10)))
        y = core.sinc(x)
        
        #plt.figure()
        #plt.plot(y.numpy(), c="r", label="output")
        #plt.plot(x.numpy(), c="b", label="input")
        #plt.legend()

    def test_sinc_impulse_response(self):
        batch_size = 2
        sr = 16000
        fc = torch.tensor([80, 100, 1000])
        fc = fc.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        
        ir = core.sinc_impulse_response(cutoff_frequency=fc, sample_rate=sr)
        
        #print(ir.shape)
        #plt.figure()
        #for i in range(ir.shape[1]):
        #    plt.plot(ir[0, i].numpy())
        #    plt.yscale('log')
        #plt.title("test_sinc_impulse_response")
        #plt.show()

    def test_sinc_filter(self):
        sample_rate = 16000
        noise = np.random.uniform(-0.5, 0.5, [1, sample_rate *4])
        noise = torch.as_tensor(noise, dtype=torch.float32)
        f_cutoff = np.linspace(0., sample_rate/2, 200)[np.newaxis, :, np.newaxis]
        f_cutoff = torch.as_tensor(f_cutoff, dtype=torch.float32)
        
        y = core.sinc_filter(audio=noise, cutoff_frequency=f_cutoff,
                             sample_rate=sample_rate, window_size=128)
        
        #plot_spec(y, sample_rate, "test_sinc_filter")
    
    def test_fft_convolve(self):
        sample_rate = 16000
        noise = np.random.uniform(-0.5, 0.5, [1, sample_rate *4])
        noise = torch.as_tensor(noise, dtype=torch.float32)
        f_cutoff = np.linspace(0., 1.0, 200)[np.newaxis, :, np.newaxis]
        f_cutoff = torch.as_tensor(f_cutoff, dtype=torch.float32)
        
        ir = core.sinc_impulse_response(f_cutoff , 2048)
        y = core.fft_convolve(noise, ir)
        
        #plot_spec(y, sample_rate, "test_fft_convolve")
    
    def test_time_domain_FIR(self):
        sample_rate = 16000
        batch_size = 2
        n_filters = 3
        N = 65 # Filter-order is N - 1

        # Generate an input signal of length L
        duration = 1 # sec
        L = sample_rate * duration
        input_signal = torch.rand((batch_size, n_filters, L)) * 2 - 1   

        # Make time varying fc of shape [batch_size, n_time, 1].
        n_frames = 100
        fcs = torch.linspace(0, sample_rate/2, n_frames)
        fcs = fcs.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        
        # Get lowpass impulse responses.
        hs = core.sinc_impulse_response(cutoff_frequency=fcs,
                                        window_size=N-1,
                                        sample_rate=sample_rate)
        
        # Build one batch of impulse responses.
        h11 = hs[0, 1].unsqueeze(0)
        h12 = hs[0, 10].unsqueeze(0)
        h13 = hs[0, 20].unsqueeze(0)
        h21 = hs[0, 30].unsqueeze(0)
        h22 = hs[0, 40].unsqueeze(0)
        h23 = hs[0, 60].unsqueeze(0)
        h1 = torch.cat((h11, h12, h13)).unsqueeze(0)
        h2 = torch.cat((h21, h22, h23)).unsqueeze(0)
        h = torch.cat((h1, h2), dim=0)
        
        # Create the FIR filters with the impulse responses.
        fir_filter = core.TimeDomainFIR(h)
        
        sig_len = input_signal.shape[-1]
        # We do not care about the last N - 1 samples of the convolution.
        y = torch.zeros_like(input_signal)
        for sample in range(sig_len):
            x_n = input_signal[..., sample]
            y[..., sample] = fir_filter(x_n)
        
        #print(y.shape)
        for batch in range(batch_size):
            for filter_n in range(n_filters):
                plot_spec(y[batch, filter_n], sample_rate,
                          "test_time_domain_FIR" + str(batch) + \
                          " filter: " + str(filter_n))
        
    def test_time_domain_FIR_differentiability(self):
        batch_size = 2
        n_sources = 3
        N = 65
        fc = 2500
        sr = 16000

        fcs = torch.ones((batch_size, n_sources, 1)) * fc
        hs = core.sinc_impulse_response(cutoff_frequency=fcs,
                                        window_size=N-1,
                                        sample_rate=sr)
        
        filt = core.TimeDomainFIR(hs)
        
        dur = 0.02 # sec
        sig_len = int(dur * sr)
        x = torch.rand((batch_size, n_sources, sig_len)) * 2 - 1

        for i in range(sig_len):
            x_in = x[..., i]
            # Zeroing out the gradient
            if x_in.grad is not None:
                x_in.grad.zero_()
            if fcs.grad is not None:
                fcs.grad.zero_()
            # Detach from current graph.
            filt.detach()
            
            # Set parameter to calculate gradient
            x_in.requires_grad = True
            fcs.requires_grad = True
            
            y = filt(x_in)
            
            # Dummy cost function
            error = y.sum()
            error.backward()

if __name__ == '__main__':
    test = TestCore()
    
    unittest.main()
    
    #test.test_simple_highpass_differentiability()
    #test.test_sinc_filter()
    #test.test_fft_convolve()
    #test.test_time_domain_FIR()
    #test.test_time_domain_FIR_differentiability()
    
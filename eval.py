import os
import pickle
import json
import argparse
import tqdm

import torch
import numpy as np
import pandas as pd
import librosa as lb

import data
import models
import utils
import evaluation_metrics as em
import ddsp.spectral_ops


torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str)
parser.add_argument('--test-set', type=str, default='El Rossinyol', choices=['CSD'])
parser.add_argument('--f0-from-mix', action='store_true', default=False)

args, _ = parser.parse_known_args()

tag = args.tag

is_u_net = tag[:4] == 'unet'

parser.add_argument('--eval-tag', type=str, default=tag)
args, _ = parser.parse_known_args()

# single voice f0 tracker used (monophonic pitch tracker)
f0_add_on = 'sf0'
# multi voice f0 tracker used (polyphonic pitch tracker)
if args.f0_from_mix: f0_add_on = 'mf0'

if args.test_set == 'CSD': test_set_add_on = 'CSD'

path_to_save_results = 'evaluation/{}/eval_results_{}_{}'.format(args.eval_tag, f0_add_on, test_set_add_on)
if not os.path.isdir(path_to_save_results):
    os.makedirs(path_to_save_results, exist_ok=True)

if is_u_net: path_to_save_results_masking = path_to_save_results
else:
    path_to_save_results_masking = 'evaluation/{}/eval_results_{}_{}'.format(args.eval_tag + '_masking', f0_add_on, test_set_add_on)
    if not os.path.isdir(path_to_save_results_masking):
        os.makedirs(path_to_save_results_masking, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trained_model, model_args = utils.load_model(tag, device, return_args=True)
trained_model.return_synth_params = False
trained_model.return_sources=True

voices = model_args['voices'] if 'voices' in model_args.keys() else 'satb'
original_cunet = model_args['original_cu_net'] if 'original_cu_net' in model_args.keys() else False
f0_cuesta = args.f0_from_mix


if args.test_set == 'CSD':
    el_rossinyol = data.CSD(song_name='El Rossinyol', example_length=model_args['example_length'], allowed_voices=voices,
                        return_name=True, n_sources=model_args['n_sources'], singer_nb=[2], random_mixes=False,
                        f0_from_mix=f0_cuesta, cunet_original=original_cunet)

    locus_iste = data.CSD(song_name='Locus Iste', example_length=model_args['example_length'], allowed_voices=voices,
                         return_name=True, n_sources=model_args['n_sources'], singer_nb=[3], random_mixes=False,
                         f0_from_mix=f0_cuesta, cunet_original=original_cunet)

    nino_dios = data.CSD(song_name='Nino Dios', example_length=model_args['example_length'], allowed_voices=voices,
                     return_name=True, n_sources=model_args['n_sources'], singer_nb=[4], random_mixes=False,
                     f0_from_mix=f0_cuesta, cunet_original=original_cunet)

    test_set = torch.utils.data.ConcatDataset([el_rossinyol, locus_iste, nino_dios])


eval_results = pd.DataFrame({'mix_name': [], 'eval_seed': [], 'voice': [], 'eval_frame': [], 'sp_SNR': [], 'sp_SI-SNR': [],
                             'mel_cep_dist': []})

eval_results_masking = pd.DataFrame({'mix_name': [], 'eval_seed': [], 'voice': [], 'eval_frame': [], 'sp_SNR': [], 'sp_SI-SNR': [],
                                     'SI-SDR': [], 'mel_cep_dist': []})

pd.set_option("display.max_rows", None, "display.max_columns", None)

if is_u_net: n_seeds = 1
else: n_seeds = 5

for seed in range(n_seeds):
    torch.manual_seed(seed)
    rng_state_torch = torch.get_rng_state()

    #for d in data_loader:
    pbar = tqdm.tqdm(test_set)
    for d in pbar:
        mix = d[0].to(device)
        f0_hz = d[1].to(device)
        target_sources = d[2].to(device)
        name = d[3]
        voices = d[4]

        mix = mix.unsqueeze(0)
        target_sources = target_sources.unsqueeze(0)
        f0_hz = f0_hz[None, :, :]

        batch_size, n_samples, n_sources = target_sources.shape

        n_fft_metrics = 512
        overlap_metrics = 0.5

        # reset rng state so that each example gets the same state
        torch.random.set_rng_state(rng_state_torch)

        with torch.no_grad():

            if is_u_net:
                n_hop = int(trained_model.n_fft - trained_model.overlap * trained_model.n_fft)
                estimated_sources = utils.masking_unets_softmasks(trained_model, mix, f0_hz, n_sources,
                                                                  trained_model.n_fft, n_hop)
                # estimated_sources = utils.masking_unets2(trained_model, mix, f0_hz, n_sources,
                #                                          trained_model.n_fft, n_hop)

                source_estimates = torch.tensor(estimated_sources, device=device, dtype=torch.float32).unsqueeze(0)  # [batch_size, n_source, n_samples]
                source_estimates_masking = source_estimates.reshape((batch_size * n_sources, n_samples))

            else:
                mix_estimate, source_estimates = trained_model(mix, f0_hz)

                # [batch_size * n_sources, n_samples]
                source_estimates_masking = utils.masking_from_synth_signals_torch(mix, source_estimates, n_fft=2048, n_hop=256)

            target_sources = target_sources.transpose(1, 2)  # [batch_size, n_sources, n_samples]
            target_sources = target_sources.reshape((batch_size * n_sources, n_samples))
            source_estimates = source_estimates.reshape((batch_size * n_sources, n_samples))

            # compute spectral SNR masking
            si_snr_masking = em.spectral_si_snr(target_sources, source_estimates_masking, fft_size=n_fft_metrics, overlap=overlap_metrics)
            si_snr_masking = si_snr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute spectral SNR masking
            snr_masking = em.spectral_snr(target_sources, source_estimates_masking, fft_size=n_fft_metrics, overlap=overlap_metrics)
            snr_masking = snr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute mel cepstral distance masking
            mcd_masking = em.mel_cepstral_distance(target_sources, source_estimates_masking, fft_size=n_fft_metrics, overlap=overlap_metrics, device=device)
            mcd_masking = mcd_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute SI-SDR masking
            si_sdr_masking = em.si_sdr(target_sources, source_estimates_masking)
            si_sdr_masking = si_sdr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute spectral SNR
            snr = em.spectral_snr(target_sources, source_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics)
            snr = snr.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute spectral SI-SNR
            si_snr = em.spectral_si_snr(target_sources, source_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics)
            si_snr = si_snr.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute mel cepstral distance
            mcd = em.mel_cepstral_distance(target_sources, source_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics, device=device)
            mcd = mcd.reshape((batch_size, n_sources, -1)).cpu().numpy()
            
            # compute SI-SDR
            si_sdr = em.si_sdr(target_sources, source_estimates)
            si_sdr = si_sdr.reshape((batch_size, n_sources, -1)).cpu().numpy()

        n_eval_frames = snr.shape[-1]

        # mix_names = [n for n in name for _ in range(n_sources * n_eval_frames)]
        mix_names = [name for _ in range(n_sources * n_eval_frames)]
        # voice = [v for b in range(batch_size) for v in voices[b] for _ in range(n_eval_frames)]
        voice = [v for v in voices for _ in range(n_eval_frames)]
        eval_frame = [f for _ in range(n_sources * batch_size) for f in range(n_eval_frames)]
        seed_results = [seed] * len(eval_frame)

        si_sdr_results_masking = [si_sdr_masking[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
        snr_results_masking = [snr_masking[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
        si_snr_results_masking = [si_snr_masking[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
        mcd_results_masking = [mcd_masking[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]

        batch_results_masking = pd.DataFrame({'mix_name': mix_names,
                                              'eval_seed': seed_results,
                                              'voice': voice, 
                                              'eval_frame': eval_frame,
                                              'sp_SNR': snr_results_masking,
                                              'sp_SI-SNR': si_snr_results_masking,
                                              'SI-SDR': si_sdr_results_masking,
                                              'mel_cep_dist': mcd_results_masking})
        eval_results_masking = eval_results_masking.append(batch_results_masking, ignore_index=True)

        snr_results = [snr[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
        si_snr_results = [si_snr[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
        mcd_results = [mcd[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
        si_sdr_results = [si_sdr[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]

        batch_results = pd.DataFrame({'mix_name': mix_names,
                                      'eval_seed': seed_results,
                                      'voice': voice,
                                      'eval_frame': eval_frame,
                                      'sp_SNR': snr_results,
                                      'sp_SI-SNR': si_snr_results,
                                      'SI-SDR': si_sdr_results,
                                      'mel_cep_dist': mcd_results})
        eval_results = eval_results.append(batch_results, ignore_index=True)


# save data frame with all results
if not is_u_net: eval_results.to_pickle(path_to_save_results + '/all_results.pandas')

# compute mean, median, std over all voices and mixes and eval_frames
means = eval_results.mean(axis=0, skipna=True, numeric_only=True)
medians = eval_results.median(axis=0, skipna=True, numeric_only=True)
stds = eval_results.std(axis=0, skipna=True, numeric_only=True)

print(tag)
print('SI-SDR:', 'mean', means['SI-SDR'], 'median', medians['SI-SDR'], 'std', stds['SI-SDR'])
print('sp_SNR:', 'mean', means['sp_SNR'], 'median', medians['sp_SNR'], 'std', stds['sp_SNR'])
print('sp_SI-SNR', 'mean', means['sp_SI-SNR'], 'median', medians['sp_SI-SNR'], 'std', stds['sp_SI-SNR'])
print('mel cepstral distance:', 'mean', means['mel_cep_dist'], 'median', medians['mel_cep_dist'], 'std', stds['mel_cep_dist'])

eval_results_masking.to_pickle(path_to_save_results_masking + '/all_results.pandas')
means_masking = eval_results_masking.mean(axis=0, skipna=True, numeric_only=True)
medians_masking = eval_results_masking.median(axis=0, skipna=True, numeric_only=True)
stds_masking = eval_results_masking.std(axis=0, skipna=True, numeric_only=True)

print(tag + '_masking')
print('SI-SDR:', 'mean', means_masking['SI-SDR'], 'median', medians_masking['SI-SDR'], 'std', stds_masking['SI-SDR'])
print('sp_SNR:', 'mean', means_masking['sp_SNR'], 'median', medians_masking['sp_SNR'], 'std', stds_masking['sp_SNR'])
print('sp_SI-SNR', 'mean', means_masking['sp_SI-SNR'], 'median', medians_masking['sp_SI-SNR'], 'std', stds_masking['sp_SI-SNR'])
print('mel cepstral distance:', 'mean', means_masking['mel_cep_dist'], 'median', medians_masking['mel_cep_dist'], 'std', stds_masking['mel_cep_dist'])


# Calculate metrics and save them to
# evaluation/model_tag/eval_results_f0_variant/all_results.pandas
#
# Baselines are saved in evaluation/model_tag_baseline_type/

import os
import pickle
import json
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import pandas as pd
import librosa as lb

import data
import models
import utils
import evaluation_metrics as em
import ddsp.spectral_ops


def irm(mix, target_sources):
    """
    Ideal Ratio Mask (Oracle filtering) as upper baseline.
    Masking the mix signal with a mask obtained from the target sources.
    The phase of the estimated source is taken from the mix signal.
    
    Args:
        mix: torch.Tensor of shape [batch_size, n_samples], Mixture signal
        target_sources: torch.Tensor of shape [batch_size, n_sources, n_samples]
            target sources that form the ideal mask.
    
    Returns:
        irm_estimates: torch.Tensor of shape [batch_size, n_sources, n_samples]
            sources obtained by ideal ratio mask filtering (Oracle filtering)
    """
    eps = 1e-10
    
    n_fft = 2048
    hop_size = 256
    
    batch_size, n_sources, n_samples = target_sources.shape
    
    mix_stft = torch.stft(mix, n_fft=n_fft, hop_length=hop_size)
    X_mix = torch.view_as_complex(mix_stft).repeat(n_sources, 1, 1)
    
    target_sources = torch.reshape(target_sources, (batch_size * n_sources, n_samples))
    targets_stft = torch.stft(target_sources, n_fft=n_fft, hop_length=hop_size)
    X_targets = torch.view_as_complex(targets_stft)
    
    ideal_ratio_masks = X_targets.abs() / (eps + X_mix.abs())
    
    X_irm_estimates = X_mix * ideal_ratio_masks
    irm_estimates = torch.istft(torch.view_as_real(X_irm_estimates), n_fft=n_fft, hop_length=hop_size)
    return torch.reshape(irm_estimates, (batch_size, n_sources, n_samples))

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str)
parser.add_argument('--test-set', type=str, default='El Rossinyol', choices=['CSD', 'Guitarset'])
parser.add_argument('--f0-from-mix', action='store_true', default=False)
parser.add_argument('--no-baselines', action='store_true', default=False, help= \
                    "Do NOT calculate baseline metrics. The upper basline is the" \
                    "Ideal Ratio Mask and the lower baseline is white noise.")
parser.add_argument('--which', type=str, default='best', choices=['best', 'last'],
                    help="Which model to load: the best model on validation or the last trained one.")
args, _ = parser.parse_known_args()

tag = args.tag

is_u_net = tag[:4] == 'unet'

baselines = not args.no_baselines

parser.add_argument('--eval-tag', type=str, default=tag)
args, _ = parser.parse_known_args()

# single voice f0 tracker used (monophonic pitch tracker)
f0_add_on = 'sf0'
# multi voice f0 tracker used (polyphonic pitch tracker)
if args.f0_from_mix: f0_add_on = 'mf0'

if args.test_set == 'CSD': test_set_add_on = 'CSD'
if args.test_set == 'Guitarset': test_set_add_on = 'Guitarset'

path_to_save_results = 'evaluation/{}/eval_results_{}_{}'.format(args.eval_tag, f0_add_on, test_set_add_on)
if not os.path.isdir(path_to_save_results):
    os.makedirs(path_to_save_results, exist_ok=True)

if is_u_net: path_to_save_results_masking = path_to_save_results
else:
    path_to_save_results_masking = 'evaluation/{}/eval_results_{}_{}'.format(args.eval_tag + '_masking', f0_add_on, test_set_add_on)
    if not os.path.isdir(path_to_save_results_masking):
        os.makedirs(path_to_save_results_masking, exist_ok=True)

if baselines:
    path_to_save_results_noise = f"evaluation/{args.eval_tag}_noise/eval_results_{f0_add_on}_{test_set_add_on}"
    if not os.path.isdir(path_to_save_results_noise):
        os.makedirs(path_to_save_results_noise, exist_ok=True)
    path_to_save_results_IRM = f"evaluation/{args.eval_tag}_IRM/eval_results_{f0_add_on}_{test_set_add_on}"
    if not os.path.isdir(path_to_save_results_IRM):
        os.makedirs(path_to_save_results_IRM, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trained_model, model_args = utils.load_model(tag, args.which, device, return_args=True)
trained_model.return_synth_params = False
trained_model.return_sources = True


voices = model_args['voices'] if 'voices' in model_args.keys() else 'satb'
original_cunet = model_args['original_cu_net'] if 'original_cu_net' in model_args.keys() else False
n_sources = model_args['n_sources'] if 'n_sources' in model_args.keys() else None
f0_cuesta = args.f0_from_mix


# CSD Dataset
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


# Guitarset Dataset
n_test_files = None
if args.test_set == "Guitarset":
    n_files_per_style_genre = model_args['n_files_per_style_genre']
    valid_split = model_args['valid_split']
    n_train_files = int((1 - valid_split) * n_files_per_style_genre)
    n_valid_files = int(valid_split * n_files_per_style_genre)
    # Use same amount of test data as validation data.
    # We use the next unused files of the Dataset.
    if n_test_files == None:
        n_test_files = n_valid_files
    
    test_set = data.Guitarset(
        batch_size=model_args['batch_size'], # Must be same batch size as in training.
        dataset_range=(n_train_files + n_valid_files,
                       n_train_files + n_valid_files + n_test_files),
        style=model_args['style'],
        genres=model_args['genres'],
        allowed_strings=model_args['allowed_strings'],
        shuffle_files=False,
        conf_threshold=model_args['confidence_threshold'],
        example_length=model_args['example_length'],
        return_name=True,
        f0_from_mix=f0_cuesta,
        cunet_original=False,
        file_list=False,
        normalize_mix=model_args['normalize_mix'],
        normalize_sources=model_args['normalize_sources'])


eval_results = pd.DataFrame({'mix_name': [], 'eval_seed': [], 'voice': [], 'eval_frame': [], 'sp_SNR': [], 'sp_SI-SNR': [],
                             'SI-SDR': [], 'mel_cep_dist': []})

eval_results_masking = pd.DataFrame({'mix_name': [], 'eval_seed': [], 'voice': [], 'eval_frame': [], 'sp_SNR': [], 'sp_SI-SNR': [],
                                     'SI-SDR': [], 'mel_cep_dist': []})

if baselines:
    eval_results_noise = pd.DataFrame({'mix_name': [], 'eval_seed': [], 'voice': [], 'eval_frame': [], 'sp_SNR': [], 'sp_SI-SNR': [],
                                         'SI-SDR': [], 'mel_cep_dist': []})
    
    eval_results_IRM = pd.DataFrame({'mix_name': [], 'eval_seed': [], 'voice': [], 'eval_frame': [], 'sp_SNR': [], 'sp_SI-SNR': [],
                                         'SI-SDR': [], 'mel_cep_dist': []})

pd.set_option("display.max_rows", None, "display.max_columns", None)

if is_u_net or args.test_set == "Guitarset": n_seeds = 1
else: n_seeds = 5

for seed in range(n_seeds):
    torch.manual_seed(seed)
    rng_state_torch = torch.get_rng_state()

    if args.test_set == "CSD" or is_u_net:
        pbar = tqdm.tqdm(test_set)
    elif args.test_set == "Guitarset":
        # Guitarset does batching on its own.
        pbar = tqdm.tqdm(DataLoader(test_set, batch_size=None, shuffle=False))
    else:
        pbar = tqdm.tqdm(DataLoader(test_set, batch_size=1, shuffle=False))
    
    for d in pbar:
        mix = d[0].to(device)
        f0_hz = d[1].to(device)
        target_sources = d[2].to(device)
        name = d[3]
        voices = d[4]
        
        if args.test_set == "Guitarset" and test_set.batch_size == 1:
            name = name[0]
        if args.test_set != "Guitarset":
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

            # baselines
            if baselines:
                noise_estimates = torch.randn_like(target_sources) * 2 - 1
                irm_estimates = irm(mix, torch.transpose(target_sources, 1, 2))

            target_sources = target_sources.transpose(1, 2)  # [batch_size, n_sources, n_samples]
            target_sources = target_sources.reshape((batch_size * n_sources, n_samples))
            source_estimates = source_estimates.reshape((batch_size * n_sources, n_samples))
            if baselines:
                noise_estimates = noise_estimates.reshape((batch_size * n_sources, n_samples))
                irm_estimates = irm_estimates.reshape((batch_size * n_sources, n_samples))

            # --- Masking metrics ---

            # compute SNR masking
            snr_masking = em.spectral_snr(target_sources, source_estimates_masking, fft_size=n_fft_metrics, overlap=overlap_metrics)
            snr_masking = snr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute spectral SI-SNR masking
            si_snr_masking = em.spectral_si_snr(target_sources, source_estimates_masking, fft_size=n_fft_metrics, overlap=overlap_metrics)
            si_snr_masking = si_snr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute mel cepstral distance masking
            mcd_masking = em.mel_cepstral_distance(target_sources, source_estimates_masking, fft_size=n_fft_metrics, overlap=overlap_metrics, device=device)
            mcd_masking = mcd_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # compute SI-SDR masking
            si_sdr_masking = em.si_sdr(target_sources, source_estimates_masking)
            si_sdr_masking = si_sdr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            
            # --- Synthesized metrics ---
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
            
            if baselines:
                # --- Noise metrics ---
                # compute spectral SNR
                snr_noise = em.spectral_snr(target_sources, noise_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics)
                snr_noise = snr_noise.reshape((batch_size, n_sources, -1)).cpu().numpy()

                # compute spectral SI-SNR
                si_snr_noise = em.spectral_si_snr(target_sources, noise_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics)
                si_snr_noise = si_snr_noise.reshape((batch_size, n_sources, -1)).cpu().numpy()

                # compute mel cepstral distance
                mcd_noise = em.mel_cepstral_distance(target_sources, noise_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics, device=device)
                mcd_noise = mcd_noise.reshape((batch_size, n_sources, -1)).cpu().numpy()
                
                # compute SI-SDR
                si_sdr_noise = em.si_sdr(target_sources, noise_estimates)
                si_sdr_noise = si_sdr_noise.reshape((batch_size, n_sources, -1)).cpu().numpy()
                
                
                # --- IRM metrics ---
                # compute spectral SNR
                snr_IRM = em.spectral_snr(target_sources, irm_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics)
                snr_IRM = snr_IRM.reshape((batch_size, n_sources, -1)).cpu().numpy()

                # compute spectral SI-SNR
                si_snr_IRM = em.spectral_si_snr(target_sources, irm_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics)
                si_snr_IRM = si_snr_IRM.reshape((batch_size, n_sources, -1)).cpu().numpy()

                # compute mel cepstral distance
                mcd_IRM = em.mel_cepstral_distance(target_sources, irm_estimates, fft_size=n_fft_metrics, overlap=overlap_metrics, device=device)
                mcd_IRM = mcd_IRM.reshape((batch_size, n_sources, -1)).cpu().numpy()
                
                # compute SI-SDR
                si_sdr_IRM = em.si_sdr(target_sources, irm_estimates)
                si_sdr_IRM = si_sdr_IRM.reshape((batch_size, n_sources, -1)).cpu().numpy()

        n_eval_frames = snr.shape[-1]

        if batch_size > 1:
            mix_names = [n for n in name for _ in range(n_sources * n_eval_frames)]
        else:
            mix_names = [name for _ in range(n_sources * n_eval_frames)]
        #voice = [v for b in range(batch_size) for v in voices[b] for _ in range(n_eval_frames)]
        voice = [v for v in voices for _ in range(n_eval_frames)]
        if args.test_set == "Guitarset":
            voice = batch_size * voice
        
        
        eval_frame = [f for _ in range(n_sources * batch_size) for f in range(n_eval_frames)]
        seed_results = [seed] * len(eval_frame)

        # --- Masking metrics ---
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
        
        # --- Synthesized metrics ---
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
        
        if baselines:
            # --- Noise metrics ---
            snr_results_noise = [snr_noise[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
            si_snr_results_noise = [si_snr_noise[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
            mcd_results_noise = [mcd_noise[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
            si_sdr_results_noise = [si_sdr_noise[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]

            batch_results_noise = pd.DataFrame({'mix_name': mix_names,
                                          'eval_seed': seed_results,
                                          'voice': voice,
                                          'eval_frame': eval_frame,
                                          'sp_SNR': snr_results_noise,
                                          'sp_SI-SNR': si_snr_results_noise,
                                          'SI-SDR': si_sdr_results_noise,
                                          'mel_cep_dist': mcd_results_noise})
            eval_results_noise = eval_results_noise.append(batch_results_noise, ignore_index=True)
            
            # --- IRM metrics ---
            snr_results_IRM = [snr_IRM[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
            si_snr_results_IRM = [si_snr_IRM[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
            mcd_results_IRM = [mcd_IRM[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]
            si_sdr_results_IRM = [si_sdr_IRM[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]

            batch_results_IRM = pd.DataFrame({'mix_name': mix_names,
                                          'eval_seed': seed_results,
                                          'voice': voice,
                                          'eval_frame': eval_frame,
                                          'sp_SNR': snr_results_IRM,
                                          'sp_SI-SNR': si_snr_results_IRM,
                                          'SI-SDR': si_sdr_results_IRM,
                                          'mel_cep_dist': mcd_results_IRM})
            eval_results_IRM = eval_results_IRM.append(batch_results_IRM, ignore_index=True)


# save data frame with all results
if not is_u_net: eval_results.to_pickle(path_to_save_results + '/all_results.pandas')

# --- Synthesized metrics ---
# compute mean, median, std over all voices and mixes and eval_frames
means = eval_results.mean(axis=0, skipna=True, numeric_only=True)
medians = eval_results.median(axis=0, skipna=True, numeric_only=True)
stds = eval_results.std(axis=0, skipna=True, numeric_only=True)

print()
print(tag + ":")
print('SI-SDR:', 'mean', means['SI-SDR'], 'median', medians['SI-SDR'], 'std', stds['SI-SDR'])
print('sp_SNR:', 'mean', means['sp_SNR'], 'median', medians['sp_SNR'], 'std', stds['sp_SNR'])
print('sp_SI-SNR', 'mean', means['sp_SI-SNR'], 'median', medians['sp_SI-SNR'], 'std', stds['sp_SI-SNR'])
print('mel cepstral distance:', 'mean', means['mel_cep_dist'], 'median', medians['mel_cep_dist'], 'std', stds['mel_cep_dist'])

# --- Masking metrics ---
eval_results_masking.to_pickle(path_to_save_results_masking + '/all_results.pandas')
means_masking = eval_results_masking.mean(axis=0, skipna=True, numeric_only=True)
medians_masking = eval_results_masking.median(axis=0, skipna=True, numeric_only=True)
stds_masking = eval_results_masking.std(axis=0, skipna=True, numeric_only=True)

print()
print(tag + '_masking:')
print('SI-SDR:', 'mean', means_masking['SI-SDR'], 'median', medians_masking['SI-SDR'], 'std', stds_masking['SI-SDR'])
print('sp_SNR:', 'mean', means_masking['sp_SNR'], 'median', medians_masking['sp_SNR'], 'std', stds_masking['sp_SNR'])
print('sp_SI-SNR', 'mean', means_masking['sp_SI-SNR'], 'median', medians_masking['sp_SI-SNR'], 'std', stds_masking['sp_SI-SNR'])
print('mel cepstral distance:', 'mean', means_masking['mel_cep_dist'], 'median', medians_masking['mel_cep_dist'], 'std', stds_masking['mel_cep_dist'])

if baselines:
    # --- Noise metrics ---
    eval_results_noise.to_pickle(path_to_save_results_noise + '/all_results.pandas')
    means_noise = eval_results_noise.mean(axis=0, skipna=True, numeric_only=True)
    medians_noise = eval_results_noise.median(axis=0, skipna=True, numeric_only=True)
    stds_noise = eval_results_noise.std(axis=0, skipna=True, numeric_only=True)
    
    print()
    print(tag + '_noise as lower baseline:')
    print('SI-SDR:', 'mean', means_noise['SI-SDR'], 'median', medians_noise['SI-SDR'], 'std', stds_noise['SI-SDR'])
    print('sp_SNR:', 'mean', means_noise['sp_SNR'], 'median', medians_noise['sp_SNR'], 'std', stds_noise['sp_SNR'])
    print('sp_SI-SNR', 'mean', means_noise['sp_SI-SNR'], 'median', medians_noise['sp_SI-SNR'], 'std', stds_noise['sp_SI-SNR'])
    print('mel cepstral distance:', 'mean', means_noise['mel_cep_dist'], 'median', medians_noise['mel_cep_dist'], 'std', stds_noise['mel_cep_dist'])

    # --- IRM metrics ---
    eval_results_IRM.to_pickle(path_to_save_results_IRM + '/all_results.pandas')
    means_IRM = eval_results_IRM.mean(axis=0, skipna=True, numeric_only=True)
    medians_IRM = eval_results_IRM.median(axis=0, skipna=True, numeric_only=True)
    stds_IRM = eval_results_IRM.std(axis=0, skipna=True, numeric_only=True)
    
    print()
    print(tag + '_IRM as upper baseline:')
    print('SI-SDR:', 'mean', means_IRM['SI-SDR'], 'median', medians_IRM['SI-SDR'], 'std', stds_IRM['SI-SDR'])
    print('sp_SNR:', 'mean', means_IRM['sp_SNR'], 'median', medians_IRM['sp_SNR'], 'std', stds_IRM['sp_SNR'])
    print('sp_SI-SNR', 'mean', means_IRM['sp_SI-SNR'], 'median', medians_IRM['sp_SI-SNR'], 'std', stds_IRM['sp_SI-SNR'])
    print('mel cepstral distance:', 'mean', means_IRM['mel_cep_dist'], 'median', medians_IRM['mel_cep_dist'], 'std', stds_IRM['mel_cep_dist'])

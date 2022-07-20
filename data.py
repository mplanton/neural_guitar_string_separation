from pathlib import Path
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import scipy.special

import glob
import argparse
import random
import torch
import torchaudio
import tqdm
import os
import pickle
import csv
import itertools
import math

import utils
import ddsp.core


def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

    if args.dataset == 'musdb':
        parser.add_argument('--is-wav', action='store_true', default=False,
                            help='loads wav instead of STEMS')
        parser.add_argument('--samples-per-track', type=int, default=64)
        parser.add_argument(
            '--source-augmentations', type=str, nargs='+',
            default=['gain', 'channelswap']
        )

        args = parser.parse_args()
        dataset_kwargs = {
            'root': args.root,
            'is_wav': args.is_wav,
            'subsets': 'train',
            'target': args.target,
            'download': args.root is None,
            'seed': args.seed
        }

        source_augmentations = Compose(
            [globals()['_augment_' + aug] for aug in args.source_augmentations]
        )

        train_dataset = MUSDBDataset(
            split='train',
            samples_per_track=args.samples_per_track,
            seq_duration=args.seq_dur,
            source_augmentations=source_augmentations,
            random_track_mix=True,
            **dataset_kwargs
        )

        valid_dataset = MUSDBDataset(
            split='valid', samples_per_track=1, seq_duration=None,
            **dataset_kwargs
        )


    elif args.dataset == 'CSD':
        parser.add_argument('--confidence-threshold', type=float, default=0.4)
        parser.add_argument('--samplerate', type=int, default=16000)
        parser.add_argument('--example-length', type=int, default=64000)
        parser.add_argument('--voices', type=str, default='satb')
        parser.add_argument('--train-song', type=str, default='Nino Dios', choices=['El Rossinyol', 'Locus Iste', 'Nino Dios'])
        parser.add_argument('--val-song', type=str, default='Locus Iste', choices=['El Rossinyol', 'Locus Iste', 'Nino Dios'])
        parser.add_argument('--f0-cuesta', action='store_true', default=False)

        args = parser.parse_args()

        train_dataset = CSD(song_name=args.train_song,
                            conf_threshold=args.confidence_threshold,
                            example_length=args.example_length,
                            allowed_voices=args.voices,
                            n_sources=args.n_sources,
                            singer_nb=[2,3,4],
                            random_mixes=True,
                            f0_from_mix=args.f0_cuesta)

        valid_dataset = CSD(song_name=args.val_song,
                            conf_threshold=args.confidence_threshold,
                            example_length=args.example_length,
                            allowed_voices=args.voices,
                            n_sources=args.n_sources,
                            singer_nb=[2,3,4],
                            random_mixes=False,
                            f0_from_mix=args.f0_cuesta)

    elif args.dataset == 'BCBQ':
        parser.add_argument('--confidence-threshold', type=float, default=0.4)
        parser.add_argument('--samplerate', type=int, default=16000)
        parser.add_argument('--example-length', type=int, default=64000)
        parser.add_argument('--voices', type=str, default='satb')
        parser.add_argument('--f0-cuesta', action='store_true', default=False)
        args = parser.parse_args()

        if args.one_song:
            train_dataset = BCBQDataSets(data_set='BC',
                                         validation_subset=False,
                                         conf_threshold=args.confidence_threshold,
                                         example_length=args.example_length,
                                         n_sources=args.n_sources,
                                         random_mixes=True,
                                         return_name=False,
                                         allowed_voices=args.voices,
                                         f0_from_mix=args.f0_cuesta,
                                         cunet_original=args.original_cu_net,
                                         one_song=True)

            valid_dataset = BCBQDataSets(data_set='BC',
                                         validation_subset=True,
                                         conf_threshold=args.confidence_threshold,
                                         example_length=args.example_length,
                                         n_sources=args.n_sources,
                                         random_mixes=False,
                                         return_name=False,
                                         allowed_voices=args.voices,
                                         f0_from_mix=args.f0_cuesta,
                                         cunet_original=args.original_cu_net,
                                         one_song=True)
        else:
            bc_train = BCBQDataSets(data_set='BC',
                                    validation_subset=False,
                                    conf_threshold=args.confidence_threshold,
                                    example_length=args.example_length,
                                    n_sources=args.n_sources,
                                    random_mixes=True,
                                    return_name=False,
                                    allowed_voices=args.voices,
                                    f0_from_mix=args.f0_cuesta,
                                    cunet_original=args.original_cu_net)

            bq_train = BCBQDataSets(data_set='BQ',
                                    validation_subset=False,
                                    conf_threshold=args.confidence_threshold,
                                    example_length=args.example_length,
                                    n_sources=args.n_sources,
                                    random_mixes=True,
                                    return_name=False,
                                    allowed_voices=args.voices,
                                    f0_from_mix=args.f0_cuesta,
                                    cunet_original=args.original_cu_net)

            bc_val = BCBQDataSets(data_set='BC',
                                    validation_subset=True,
                                    conf_threshold=args.confidence_threshold,
                                    example_length=args.example_length,
                                    n_sources=args.n_sources,
                                    random_mixes=False,
                                    return_name=False,
                                    allowed_voices=args.voices,
                                    f0_from_mix=args.f0_cuesta,
                                  cunet_original=args.original_cu_net)

            bq_val = BCBQDataSets(data_set='BQ',
                                    validation_subset=True,
                                    conf_threshold=args.confidence_threshold,
                                    example_length=args.example_length,
                                    n_sources=args.n_sources,
                                    random_mixes=False,
                                    return_name=False,
                                    allowed_voices=args.voices,
                                    f0_from_mix=args.f0_cuesta,
                                  cunet_original=args.original_cu_net)

            train_dataset = torch.utils.data.ConcatDataset([bc_train, bq_train])
            valid_dataset = torch.utils.data.ConcatDataset([bc_val, bq_val])
    
    elif args.dataset == "Guitarset":
        parser.add_argument('--n-files-per-style-genre', type=int, default=6,
                            help="Number of files per style and genre (1-36) " \
                               + "of test *and* validation set.")
        parser.add_argument('--valid-split', type=float, default=0.2,
                            help="Ratio of the validation files from all files.")
        parser.add_argument('--style', type=str, default="comp",
                            help="Playing style: 'comp' = comping, 'solo' =" +\
                                 " soloing, 'all' = comping and soloing.")
        # List usage: `python arg.py -l 1234 2345 3456 4567`
        parser.add_argument('--genres', type=str, nargs='*',
                            default=['bn', 'funk', 'jazz', 'rock', 'ss'],
                            help="List of genres.\n" +\
                            "'bn', 'funk', 'jazz', 'rock', 'ss'\n" +\
                            "bn = Bossa Nova, ss = Singer Songwriter")
        parser.add_argument('--strings', type=int, action='store', nargs='*',
                            default=[1, 2, 3, 4, 5, 6], help="List of strings " +\
                            "to use as source signals (from high to low).\n" +\
                            "1 = higher E, 2 = B, 3 = G, 4 = D, 5 = A, 6 = lower E")
        parser.add_argument("--shuffle-files", type=bool, default=True, help=\
                            "If True, the training set file order is randomized.")
        parser.add_argument('--confidence-threshold', type=float, default=0.4)
        parser.add_argument('--example-length', type=int, default=64000)
        parser.add_argument('--f0-cuesta', action='store_true', default=False)
        args = parser.parse_args()
        
        n_train_files = int((1 - args.valid_split) * args.n_files_per_style_genre)
        n_valid_files = int(args.valid_split * args.n_files_per_style_genre)
        
        train_dataset = Guitarset(
            dataset_range=(0, n_train_files),
            style=args.style,
            genres=args.genres,
            allowed_strings=args.strings,
            shuffle_files=args.shuffle_files,
            conf_threshold=args.confidence_threshold,
            example_length=args.example_length,
            return_name=False,
            f0_from_mix=args.f0_cuesta,
            cunet_original=False)
        
        valid_dataset = Guitarset(
            dataset_range=(n_train_files, n_train_files + n_valid_files),
            style=args.style,
            genres=args.genres,
            allowed_strings=args.strings,
            shuffle_files=False,
            conf_threshold=args.confidence_threshold,
            example_length=args.example_length,
            return_name=False,
            f0_from_mix=args.f0_cuesta,
            cunet_original=False)
        
    return train_dataset, valid_dataset, args



class CSD(torch.utils.data.Dataset):

    def __init__(self, song_name: str, conf_threshold=0.4, example_length=64000, allowed_voices='satb',
                 return_name=False, n_sources=2, singer_nb=[2, 3, 4], random_mixes=False, f0_from_mix=False,
                 plus_one_f0_frame=False, cunet_original=False):
        """

        Args:
            song_name: str, must be one of ['El Rossinyol', 'Locus Iste', 'Nino Dios']
            conf_threshold: float, threshold on CREPE confidence value to differentiate between voiced/unvoiced frames
            example_length: int, length of the audio examples in samples
            return_name: if True, the names of the audio examples are returned (composed of title and singer_ids)
            n_sources: int, number of source to be mixed, must be in [1, 4]
            singer_nb: list of int in [1, 2, 3, 4], numbers that specify which singers should be used,
                e.g. singer 2, singer 3, and singer 4. The selection is valid for each voice group (SATB)
            random_mixes: bool, if True, time-sections, singers, and voices will be randomly chosen each epoch
                as means of data augmentation (should only be used for training data). If False, a deterministic
                set of mixes is provided that is the same at every call (for validation and test data).
        """

        self.song_name = song_name
        self.conf_threshold = conf_threshold
        self.example_length = example_length
        self.return_name = return_name
        self.n_sources = n_sources
        self.sample_rate = 16000
        self.singer_nb = singer_nb
        self.random_mixes = random_mixes
        self.f0_from_mix = f0_from_mix
        self.plus_one_f0_frame=plus_one_f0_frame  # for NMF
        self.cunet_original = cunet_original  # add some f0 frames to match representation in U-Net

        assert n_sources <= len(allowed_voices), 'number of sources ({}) is higher than ' \
                                                'allowed voiced to sample from ({})'.format(n_sources, len(allowed_voices))
        voices_dict = {'s': 0, 'a': 1, 't': 2, 'b': 3}
        self.voice_choices = [voices_dict[v] for v in allowed_voices]

        # song length in seconds
        if song_name == 'El Rossinyol': self.total_audio_length = 134; self.voice_ids = ['Soprano', 'ContraAlt','Tenor', 'Bajos']
        elif song_name == 'Locus Iste': self.total_audio_length = 190; self.voice_ids = ['Soprano', 'ContraAlt','tenor', 'Bajos']
        elif song_name == 'Nino Dios': self.total_audio_length = 103; self.voice_ids = ['Soprano', 'ContraAlt','tenor', 'Bajos']

        self.audio_files = sorted(glob.glob('../Datasets/ChoralSingingDataset/{}/audio_16kHz/*.wav'.format(song_name)))
        self.crepe_dir = '../Datasets/ChoralSingingDataset/{}/crepe_f0_center'.format(song_name)

        f0_cuesta_dir = '../Datasets/ChoralSingingDataset/{}/mixtures_{}_sources/mf0_cuesta_processed/*.pt'.format(song_name, n_sources)
        self.f0_cuesta_files = sorted(list(glob.glob(f0_cuesta_dir)))

        if not random_mixes:
            # number of non-overlapping excerpts
            n_excerpts = self.total_audio_length * self.sample_rate // self.example_length
            excerpt_idx = [i for i in range(1, n_excerpts)]

            # possible combinations of the SATB voices
            voice_combinations = list(itertools.combinations(self.voice_choices, r=n_sources))

            # possible combinations of singers
            singer_combinations = list(itertools.combinations_with_replacement([idx - 1 for idx in singer_nb], r=n_sources))

            # make list of all possible combinations
            self.examples = list(itertools.product(excerpt_idx, voice_combinations, singer_combinations))
            self.n_examples = len(self.examples)

    def __len__(self):
        if self.random_mixes: return 1600
        else: return self.n_examples

    def __getitem__(self, idx):

        if self.random_mixes:
            # sample as many voices as specified by n_sources
            if self.n_sources == 4:
                voice_indices = torch.tensor([0, 1, 2, 3])
            elif self.n_sources < 4:
                # sample voice indices from [0, 3] without replacement
                probabilities = torch.zeros((4,))
                for idx in self.voice_choices: probabilities[idx] = 1
                voice_indices = torch.multinomial(probabilities, num_samples=self.n_sources, replacement=False)
            else:
                raise ValueError("Number of sources must be in [1, 4] but got {}.".format(self.n_sources))

            # sample a number of singer_nbs with replacement
            probabilities = torch.ones((4,))
            singer_indices = torch.multinomial(probabilities, num_samples=self.n_sources, replacement=True)

            # sample audio start time in seconds
            audio_start_seconds = torch.rand((1,)) * (self.total_audio_length-self.example_length/self.sample_rate)

        else:
            # deterministic set of examples
            # tuple of example parameters (audio excerpt id, (tuple of voices), (tuple of singer ids))
            params = self.examples[idx]
            excerpt_idx, voice_indices, singer_indices = params
            audio_start_seconds = excerpt_idx * self.example_length / self.sample_rate

        # make sure the audio start time corresponds to a frame for which f0 was estimates with CREPE
        audio_start_time = audio_start_seconds // 0.016 * 256 / self.sample_rate # seconds // crepe_hop_size [s]  * crepe_hop_size [samples] / sample_rate
        audio_length = self.example_length // 256 * 256 / self.sample_rate  # length in seconds
        crepe_start_frame = int(audio_start_time/0.016)
        crepe_end_frame = crepe_start_frame + int(audio_length / 0.016)

        if self.plus_one_f0_frame: crepe_end_frame += 1

        if self.cunet_original:
            crepe_start_frame -= 2
            crepe_end_frame += 2

        # load files (or just the required duration)
        sources_list = []
        f0_list = []
        name = self.song_name.replace(' ', '_').lower()
        contained_singer_ids = []

        for n in range(self.n_sources):

            voice = self.voice_ids[voice_indices[n]]

            audio_file = [f for f in self.audio_files if voice in f][singer_indices[n]]

            audio = utils.load_audio(audio_file, start=audio_start_time, dur=audio_length)[0, :]

            sources_list.append(audio)

            file_name = audio_file.split('/')[-1][:-4]

            confidence_file = '{}/{}_confidence.npy'.format(self.crepe_dir, file_name)
            confidence = np.load(confidence_file)[crepe_start_frame:crepe_end_frame]
            f0_file = '{}/{}_frequency.npy'.format(self.crepe_dir, file_name)
            frequency = np.load(f0_file)[crepe_start_frame:crepe_end_frame]
            frequency = np.where(confidence < self.conf_threshold, 0, frequency)

            frequency = torch.from_numpy(frequency).type(torch.float32)
            f0_list.append(frequency)

            singer_id = '_' + voice[0].replace('C', 'A') + file_name[-6:]
            contained_singer_ids.append(singer_id)
            name += '{}'.format(singer_id)

            if not self.plus_one_f0_frame and not self.cunet_original:
                assert len(audio) / 256 == len(frequency), 'audio and frequency lengths are inconsistent'

        sources = torch.stack(sources_list, dim=1)  # [n_samples, n_sources]

        if self.f0_from_mix:
            permutations = list(itertools.permutations(contained_singer_ids))
            permuted_mix_ids = [''.join(s) for s in permutations]
            f0_from_mix_file = [file for file in self.f0_cuesta_files if any([ids in file for ids in permuted_mix_ids])][0]
            f0_estimates = torch.load(f0_from_mix_file)[crepe_start_frame:crepe_end_frame, :]
            frequencies = f0_estimates
        else:
            frequencies = torch.stack(f0_list, dim=1)  # [n_frames, n_sources]

        name += '_{}'.format(np.round(audio_start_time, decimals=3))

        # mix and normalize
        mix = torch.sum(sources, dim=1)  # [n_samples]
        mix_max = mix.abs().max()
        mix = mix / mix_max
        sources = sources / mix_max  # [n_samples, n_sources]

        voices = ''.join(['satb'[x] for x in voice_indices])

        if self.return_name: return mix, frequencies, sources, name, voices
        else: return mix, frequencies, sources


class CSDSongDataset(torch.utils.data.Dataset):
    """
    A dataset class to make source separation with a single song of the Choral Singing Dataset.
    This is practical for inference.
    """
    def __init__(self, model_args, test_set='CSD', song_name="Nino Dios", example_length=64000):
        self.ds_path = "../Datasets/ChoralSingingDataset/"
        
        # choose sources
        self.voice_dict = {'b': "Bajos", 'a': "ContraAlt", 's': "Soprano", 't': "Tenor"}
        
        source_paths = []
        for v in model_args['voices']:
            voice = self.voice_dict[v]
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
        self.sources = torch.stack(sources)  # (n_sources, n_samples)
        mix = torch.sum(sources, dim=0) # (n_samples)
        mix_max = mix.abs().max()
        self.mix = mix / mix_max

    
    def __len__(self):
        return len(self.mix) // self.example_length
    
    def __getitem__(self, index):
        mix_slice = self.mix[index * self.example_length : (index+1) * self.example_length]
        freqs_slice = self.freqs[index * self.freqs_per_example : (index+1) * self.freqs_per_example]
        sources_slice = self.sources[index * self.example_length : (index+1) * self.example_length]
        return mix_slice, freqs_slice, sources_slice



class BCBQDataSets(torch.utils.data.Dataset):

    def __init__(self, data_set='BC', validation_subset=False, conf_threshold=0.4, example_length=64000, allowed_voices='satb',
                 return_name=False, n_sources=2, random_mixes=False, f0_from_mix=True, cunet_original=False, one_song=False):

        super().__init__()

        self.data_set = data_set
        self.conf_threshold = conf_threshold
        self.example_length = example_length
        self.allowed_voices = allowed_voices
        self.return_name = return_name
        self.n_sources = n_sources
        self.random_mixes = random_mixes
        self.f0_from_mix = f0_from_mix
        self.sample_rate = 16000
        self.cunet_original = cunet_original # if True, add 2 f0 values at start and end to match frame number in U-Net
        self.one_song = one_song

        assert n_sources <= len(allowed_voices), 'number of sources ({}) is higher than ' \
                                                 'allowed voiced to sample from ({})'.format(n_sources, len(allowed_voices))
        voices_dict = {'s': 0, 'a': 1, 't': 2, 'b': 3}
        self.voice_choices = [voices_dict[v] for v in allowed_voices]
        self.voice_ids = ['s', 'a', 't', 'b']

        self.audio_files = sorted(glob.glob('../Datasets/{}/audio_16kHz/*.wav'.format(data_set)))
        # file 17_BC021_part11_s_1ch.wav is empty  --> exclude 17_BC021_part11
        self.audio_files = [f for f in self.audio_files if '17_BC021_part11' not in f]
        self.crepe_dir = '../Datasets/{}/crepe_f0_center'.format(data_set)

        if data_set == 'BC':
            if one_song:
                #  use BC 1 for training and BC 2 for validation
                if validation_subset: self.audio_files = [f for f in self.audio_files if '2_BC002' in f]
                else: self.audio_files = [f for f in self.audio_files if '1_BC001' in f]
            else:
                if validation_subset: self.audio_files = self.audio_files[- (14+17)*4 :]  # only 8_BC and 9_BC
                else: self.audio_files = self.audio_files[: - (14+17)*4]  # all except 8_BC and 9_BC
        elif data_set == 'BQ':
            if one_song:
                raise NotImplementedError
            else:
                if validation_subset: self.audio_files = self.audio_files[- (13+11)*4 :]  # only 8_BQ and 9_BQ
                else: self.audio_files = self.audio_files[: - (13+11)*4]  # all except 8_BQ and 9_BQ

        self.f0_cuesta_dir = '../Datasets/{}/mixtures_{}_sources/mf0_cuesta_processed'.format(data_set, n_sources)

        if not random_mixes:
            # number of non-overlapping excerpts
            n_wav_files = len(self.audio_files)

            self.excerpts_per_part = ((10 * self.sample_rate) // self.example_length)

            n_excerpts = (n_wav_files // 4) * self.excerpts_per_part
            excerpt_idx = [i for i in range(n_excerpts)]

            # possible combinations of the SATB voices
            voice_combinations = list(itertools.combinations(self.voice_choices, r=n_sources))

            # make list of all possible combinations
            self.examples = list(itertools.product(excerpt_idx, voice_combinations))
            self.n_examples = len(self.examples)

        else:
            # 1 epoch = going 4 times through parts and sample different voice combinations
            self.n_examples = len(self.audio_files) // 4 * 4
            if one_song: self.n_examples = self.n_examples * 3

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):

        if self.random_mixes:
            # sample as many voices as specified by n_sources
            if self.n_sources == 4:
                voice_indices = torch.tensor([0, 1, 2, 3])
            elif self.n_sources < 4:
                # sample voice indices from [0, 3] without replacement
                probabilities = torch.zeros((4,))
                for i in self.voice_choices: probabilities[i] = 1
                voice_indices = torch.multinomial(probabilities, num_samples=self.n_sources, replacement=False)
                voice_indices = sorted(voice_indices.tolist())
            else:
                raise ValueError("Number of sources must be in [1, 4] but got {}.".format(self.n_sources))

            # sample audio start time in seconds
            audio_start_seconds = torch.rand((1,)) * (10 - 0.08 - self.example_length/self.sample_rate) + 0.04
            audio_start_seconds = audio_start_seconds.numpy()[0]
            if self.one_song:
                song_part_id = self.audio_files[idx * 4 // 12].split('/')[-1][:-10]
            else:
                song_part_id = self.audio_files[idx * 4 // 4].split('/')[-1][:-10]
        else:
            # deterministic set of examples
            # tuple of example parameters (audio excerpt id, (tuple of voices))
            params = self.examples[idx]

            excerpt_idx, voice_indices = params

            song_part_id = self.audio_files[excerpt_idx//self.excerpts_per_part * 4].split('/')[-1][:-10]

            audio_start_seconds = (excerpt_idx % self.excerpts_per_part) * self.example_length / self.sample_rate + 0.04

        # make sure the audio start time corresponds to a frame for which f0 was estimates with CREPE
        audio_start_time = audio_start_seconds // 0.016 * 256 / self.sample_rate # seconds // crepe_hop_size [s]  * crepe_hop_size [samples] / sample_rate
        audio_length = self.example_length // 256 * 256 / self.sample_rate  # length in seconds
        crepe_start_frame = int(audio_start_time/0.016)
        crepe_end_frame = crepe_start_frame + int(audio_length / 0.016)

        if self.cunet_original:
            crepe_start_frame -= 2
            crepe_end_frame += 2

        # load files (or just the required duration)
        sources_list = []
        f0_list = []
        name = song_part_id + '_'

        for n in range(self.n_sources):

            voice = self.voice_ids[voice_indices[n]]
            voice = '_' + voice + '_'

            audio_file = [f for f in self.audio_files if song_part_id in f and voice in f][0]

            audio = utils.load_audio(audio_file, start=audio_start_time, dur=audio_length)[0, :]

            sources_list.append(audio)

            file_name = audio_file.split('/')[-1][:-4]

            confidence_file = '{}/{}_confidence.npy'.format(self.crepe_dir, file_name)
            confidence = np.load(confidence_file)[crepe_start_frame:crepe_end_frame]
            f0_file = '{}/{}_frequency.npy'.format(self.crepe_dir, file_name)
            frequency = np.load(f0_file)[crepe_start_frame:crepe_end_frame]
            frequency = np.where(confidence < self.conf_threshold, 0, frequency)

            frequency = torch.from_numpy(frequency).type(torch.float32)
            f0_list.append(frequency)

            name += voice[1]

            if not self.cunet_original:
                assert len(audio) / 256 == len(frequency), 'audio and frequency lengths are inconsistent'

        sources = torch.stack(sources_list, dim=1)  # [n_samples, n_sources]

        if self.f0_from_mix:
            f0_from_mix_file = self.f0_cuesta_dir + '/' + name + '.pt'
            f0_estimates = torch.load(f0_from_mix_file)[crepe_start_frame:crepe_end_frame, :]
            frequencies = f0_estimates
        else:
            frequencies = torch.stack(f0_list, dim=1)  # [n_frames, n_sources]

        name += '_{}'.format(np.round(audio_start_time, decimals=3))

        # mix and normalize
        mix = torch.sum(sources, dim=1)  # [n_samples]
        mix_max = mix.abs().max()
        mix = mix / mix_max
        sources = sources / mix_max  # [n_samples, n_sources]

        voices = ''.join(['satb'[x] for x in voice_indices])

        if self.return_name: return mix, frequencies, sources, name, voices
        else: return mix, frequencies, sources



class Guitarset(torch.utils.data.IterableDataset):
    """
    Generate `batch_size` parallel audio examples of length `n_samples`.
    The dataset groups the given audio files dynamically in batches and
    aranges them consecutively.
    The batching is done here so set `batch_size=None` in the DataLoader!
    The Dataset returns torch.tensor of shape [batch_size, n_sources, n_samples]
    The Dataset is empty when the first end of a stream of audio files is reached.
    This dataset loads examples separately from disk.
    
    Currently this does NOT support muti process loading with multiple workers!
    So make shure to set `num_workers` to 0 or 1 in the DataLoader.
    
    The Guitarset is a dataset of single string classical guitar recordings featuring
        * 5 different musical genres: Bossa Nova, Funk, Jazz, Rock, Singer Songwriter
        * 2 different playing styles
    There are 36 recordings per style and genre, which makes a total of 360
    recordings with an average duration of 30 seconds.
    
    The dataset is expected to have a sample rate of 16kHz and f0 information
    must be provided via single f0 tracking with CREPE or via multiple f0
    tracking with Cuesta.
    
    Args:
        batch_size: int, batch size of the loaded data.
        dataset_range: tuple of int, (start_track, stop_track).
            Specify the range of recordings per style and genre to use from the dataset.
            There are up to 36 recordings per style and genre available.
            This allows for specifying unique datasets, i.e. for training set and validation set,
            by using different dataset ranges.
        style: str, 'comp', 'solo' or 'all'.
            Playing styles are comping, solo playing or both styles.
        genres: list of str, list of musical genres from ['bn', 'funk', 'jazz', 'rock', 'ss'].
            bn = Bossa Nova, ss = Singer Songwriter
        allowed_strings: list of int, List of string numbers representing the sources
            (from high to low).
            1 = higher E, 2 = B, 3 = G, 4 = D, 5 = A, 6 = lower E
        shuffle_files: bool, If True, the recordings order is randomized per epoch.
        conf_threshold: float, threshold on CREPE confidence value to differentiate between voiced/unvoiced frames
        example_length: int, Length of the audio examples in samples.
        return_name: bool, If True, the names of the audio examples are returned (composed of title and string-number).
        f0_from_mix: bool, If True, use Cuesta multiple f0 tracker data, if False use
            CREPE single f0 tracker data as frequency information.
        cunet_original: bool, Set to True, if using the CU-Net original.
        file_list: list, If a list of files is present, the used files for the dataset are not specified by the `dataset_range`, but by this list.
    
    """
    def __init__(self,
                 batch_size,
                 dataset_range=(0, 36),
                 style='comp',
                 genres=['bn', 'funk', 'jazz', 'rock', 'ss'],
                 allowed_strings=[1, 2, 3, 4, 5, 6],
                 shuffle_files=True,
                 conf_threshold=0.4,
                 example_length=64000,
                 return_name=False,
                 f0_from_mix=False,
                 cunet_original=False,
                 file_list=False,
                 normalize_mix=True,
                 normalize_sources=True):
        super(Guitarset).__init__()
        
        # Check arguments
        assert len(dataset_range) == 2, "dataset_range must be a tuple of length 2!"
        assert style.lower() in ['comp', 'solo', 'all'], f"{style} is not a valid option for playing style!"
        assert len(genres) > 0, "List of musical genres must at least contain one genre!"
        assert len(allowed_strings) > 0, "List of allowed strings must at least contain one string number!"
        
        self.sample_rate = 16000
        # Audio example offset correction in milliseconds.
        # Gets added to frame start.
        self.crepe_hop_size = 256 # CREPE hopsize is 16ms in samples
        self.n_frames = math.ceil(example_length / self.crepe_hop_size)
        self.crepe_dir = "../Datasets/Guitarset/crepe_f0_center/"
        self.batch_size = batch_size
        self.dataset_range = dataset_range
        self.style = style.lower()
        self.genres = genres
        self.conf_threshold = conf_threshold
        self.example_length = example_length # Length of the examples in samples
        self.allowed_strings = sorted(allowed_strings)
        # Channel indices are reversed since string 1 is highest and 6 lowest.
        self.channel_indices = np.array(self.allowed_strings[::-1]) - 1
        self.n_sources = len(allowed_strings)
        self.f0_cuesta_dir = f"../Datasets/Guitarset/mixtures_{self.n_sources}_sources/mf0_cuesta_processed"
        self.return_name = return_name
        self.f0_from_mix = f0_from_mix
        self.cunet_original = cunet_original
        self.shuffle_files = shuffle_files
        self.file_list = file_list
        self.normalize_mix = normalize_mix
        self.normalize_sources = normalize_sources
        
        # Sort and filter file paths.
        # Paths have to be sorted first to get distinct datasets from files
        # (train, valid, test).
        audio_files = sorted(glob.glob("../Datasets/Guitarset/audio_16kHz/*.wav"))
        
        # Filter audio file paths to selection.
        if not file_list:
            if style != 'all':
                audio_files = [path for path in audio_files if style in path]
            self.audio_files = []
            for genre in self.genres:
                start = dataset_range[0]
                stop = dataset_range[1]
                self.audio_files += [path for path in audio_files if genre.lower() in path.lower()][start:stop]
        else:
            self.audio_files = self.file_list
        
        assert len(self.audio_files) >= batch_size, \
            "The number of audio files in the dataset must at least be the batch_size!" \
            + f"\nThe batch_size is {batch_size} and number of audio files is {len(self.audio_files)}."
        
        # Randomize filtered file paths.
        if self.shuffle_files == True:
            random.shuffle(self.audio_files)

        # Current audio file paths to be loaded
        self.current_paths = self.audio_files[:batch_size]
        # Index for loading new audio files
        self.path_idx = batch_size
        # Current file offsets
        self.offsets = batch_size * [0]
        

    def load_sources(self, batch):
        audio_file_path = self.current_paths[batch]
        offset = self.offsets[batch]
        audio, sr = torchaudio.load(audio_file_path, offset=offset,
                                    num_frames=self.example_length)
        return audio[self.channel_indices] # [n_sources, n_samples]


    def load_frequencies(self, batch):
        name = self.current_paths[batch].split('/')[-1].split('.')[0]
        sample_offset = self.offsets[batch]
        offset = int(sample_offset / self.crepe_hop_size)
        
        if not self.f0_from_mix:
            # Load CREPE frequency
            confidence_file = os.path.join(self.crepe_dir, name + "_confidence.npy")
            f0_file = os.path.join(self.crepe_dir, name + "_frequency.npy")
            confidences = np.load(confidence_file)
            confidences = confidences[self.channel_indices, offset : offset + self.n_frames]
            frequency = np.load(f0_file)
            frequency = frequency[self.channel_indices, offset : offset + self.n_frames]
            frequency = np.where(confidences < self.conf_threshold, 0, frequency)
            frequencies = torch.from_numpy(frequency).type(torch.float32) # [n_sources, n_frames]
        else:
            # Load cuesta frequency
            f0_from_mix_file = os.path.join(self.f0_cuesta_dir, + name + '.pt')
            frequencies = torch.load(f0_from_mix_file)[offset : offset + self.n_frames, :]
        return frequencies # [n_sources, n_frames]

    def __iter__(self):
        """Yields: mix, frequencies, sources
            mix: [batch_size, example_length] Mixture signal.
            frequencies: [batch_size, n_frames, n_sources] Sources fundamental frequencies.
            sources: [batch_size, example_length, n_sources] Source signals.
        """
        stop = False
        while stop == False:
            # Initialize batch with zeros for zero padding.
            sources = torch.zeros((self.batch_size, self.n_sources, self.example_length))
            frequencies = torch.zeros((self.batch_size, self.n_sources, self.n_frames))
            
            for batch in range(self.batch_size):
                # Load sources
                audio = self.load_sources(batch)
                audio_len = audio.shape[-1]
                sources[batch, :, : audio_len] = audio
                
                # Make mix and normalize
                mix = torch.sum(sources, dim=1)  # [batch_size, n_samples]
                if self.normalize_mix == True:
                    mix_max = mix.abs().max()
                    mix = mix / mix_max
                if self.normalize_sources == True:
                    mix_max = mix.abs().max() # TODO: Make this more sophisticated...
                    sources = sources / mix_max  # [batch_size, n_sources, n_samples]
                
                # Load frequencies
                freqs = self.load_frequencies(batch)
                freqs_len = freqs.shape[-1]
                frequencies[batch, :, : freqs_len] = freqs
                
                # House keeping
                self.offsets[batch] += audio_len
                if audio_len < self.example_length:
                    if self.path_idx >= len(self.audio_files):
                        stop = True
                        break
                    # Get new file
                    self.current_paths[batch] = self.audio_files[self.path_idx]
                    self.path_idx += 1
                    self.offsets[batch] = 0
            
            sources = torch.transpose(sources, 1, 2) # [batch_size, n_samples, n_sources]
            frequencies = torch.transpose(frequencies, 1, 2) # [batch_size, n_frames, n_sources]
            if self.return_name:
                names = [name.split('/')[-1].split('.')[0] for name in self.current_paths]
                yield mix, frequencies, sources, names, self.allowed_strings
            else:
                yield mix, frequencies, sources



class GuitarsetOld(torch.utils.data.Dataset):
    """
    The Guitarset is a dataset of single string classical guitar recordings featuring
        * 5 different musical genres: Bossa Nova, Funk, Jazz, Rock, Singer Songwriter
        * 2 different playing styles
    There are 36 recordings per style and genre, which makes a total of 360
    recordings with an average duration of 30 seconds.
    
    The dataset is expected to have a sample rate of 16kHz and f0 information
    must be provided via single f0 tracking with CREPE or via multiple f0
    tracking with Cuesta.
    
    Args:
        dataset_range: tuple of int, (start_track, stop_track).
            Specify the range of recordings per style and genre to use from the dataset.
            There are up to 36 recordings per style and genre available.
            This allows for specifying unique datasets, i.e. for training set and validation set,
            by using different dataset ranges.
        style: str, 'comp', 'solo' or 'all'.
            Playing styles are comping, solo playing or both styles.
        genres: list of str, list of musical genres from ['bn', 'funk', 'jazz', 'rock', 'ss'].
            bn = Bossa Nova, ss = Singer Songwriter
        allowed_strings: list of int, List of string numbers representing the sources
            (from high to low).
            1 = higher E, 2 = B, 3 = G, 4 = D, 5 = A, 6 = lower E
        shuffle_files: bool, If True, the recordings order is randomized per epoch.
        conf_threshold: float, threshold on CREPE confidence value to differentiate between voiced/unvoiced frames
        example_length: int, Length of the audio examples in samples.
        return_name: bool, If True, the names of the audio examples are returned (composed of title and string-number).
        f0_from_mix: bool, If True, use Cuesta multiple f0 tracker data, if False use
            CREPE single f0 tracker data as frequency information.
        cunet_original: bool, Set to True, if using the CU-Net original.
        file_list: list, If a list of files is present, the used files for the dataset are not specified by the `dataset_range`, but by this list.
    """
    def __init__(self,
                 dataset_range=(0, 36),
                 style='comp',
                 genres=['bn', 'funk', 'jazz', 'rock', 'ss'],
                 allowed_strings=[1, 2, 3, 4, 5, 6],
                 shuffle_files=True,
                 conf_threshold=0.4,
                 example_length=64000,
                 return_name=False,
                 f0_from_mix=False,
                 cunet_original=False,
                 file_list=False):
        super().__init__()
        
        # Check arguments
        assert len(dataset_range) == 2, "dataset_range must be a tuple of length 2!"
        assert style.lower() in ['comp', 'solo', 'all'], f"{style} is not a valid option for playing style!"
        assert len(genres) > 0, "List of musical genres must at least contain one genre!"
        assert len(allowed_strings) > 0, "List of allowed strings must at least contain one string number!"
        
        self.sample_rate = 16000
        self.dataset_range = dataset_range
        self.style = style.lower()
        self.genres = genres
        self.conf_threshold = conf_threshold
        self.example_length = example_length
        self.allowed_strings = sorted(allowed_strings)
        self.n_sources = len(allowed_strings)
        self.return_name = return_name
        self.f0_from_mix = f0_from_mix
        self.cunet_original = cunet_original
        self.shuffle_files = shuffle_files
        # Audio example offset correction in milliseconds.
        # Gets added to frame start.
        self.offset_correction = 0.04
        self.crepe_dir = "../Datasets/Guitarset/crepe_f0_center/"
        self.f0_cuesta_dir = f"../Datasets/Guitarset/mixtures_{self.n_sources}_sources/mf0_cuesta_processed"
        self.file_list = file_list
        
        # Sort and filter file paths.
        # Paths have to be sorted first to get distinct datasets from files
        # (train, valid, test).
        audio_files = sorted(glob.glob("../Datasets/Guitarset/audio_16kHz/*.wav"))
        
        if not file_list:
            if style != 'all':
                audio_files = [path for path in audio_files if style in path]
            
            self.audio_files = []
            for genre in self.genres:
                start = dataset_range[0]
                stop = dataset_range[1]
                self.audio_files += [path for path in audio_files if genre.lower() in path.lower()][start:stop]
        else:
            self.audio_files = self.file_list
        
        # Randomize filtered file paths.
        if self.shuffle_files == True:
            random.shuffle(self.audio_files)

        # Get lengths of audio files and check sample rate.
        # Every audio file is zero padded to the next example size.
        channels = 6
        lengths = []
        for path in self.audio_files:
            info = torchaudio.info(path)
            # Check sample rate
            assert info[0].rate == self.sample_rate, \
                f"{path} has the wrong sample rate!\n\
                It should be {self.samplerate}Hz, but is {info[0].rate}."
            # Sox backend of torchaudio gives a length, which is the total number
            # of samples across all channels.
            # All channels have the same length (multi channel recording).
            lengths.append(info[0].length // channels)
        
        # Calculate the number of examples per file and in total.
        # The example offset is corrected for the f0 data.
        offset_corrected_lengths = np.array(lengths) - (self.offset_correction * self.sample_rate)
        n_examples_per_file = np.ceil(offset_corrected_lengths / self.example_length)
        self.n_examples = int(n_examples_per_file.sum())
        
        # Build example ranges per file for dataset indexing.
        start = 0
        self.audio_example_ranges = []
        for n_examples in n_examples_per_file:
            n_examples = int(n_examples)
            self.audio_example_ranges.append(range(start, start + n_examples))
            start += n_examples

    def __len__(self):
        return self.n_examples
    
    def get_file_and_index(self, idx):
        """
        Dataset indexing.
        Every file gets zero padded to a full example length.
        This function returns the right audio file path with its local index.
        """
        # Searching through all files should be OK since we have not more than
        # 360 files.
        path_idx = None
        local_idx = None
        local_range = None # For debugging
        for i, local_range in enumerate(self.audio_example_ranges):
            path_idx = i
            if idx in local_range:
                local_idx = idx - local_range.start
                break
        audio_file_path = self.audio_files[path_idx]
        return audio_file_path, local_idx, local_range
    
    def __getitem__(self, idx):
        """
        Returns: mix, frequencies, sources
            mix: [batch_size, example_length] Mixture signal.
            frequencies: [batch_size, f0_example_length, n_sources] Sources fundamental frequencies.
            sources: [batch_size, example_length, n_sources] Source signals.
        """
        
        audio_file_path, local_idx, local_range = self.get_file_and_index(idx)
        
        #DBG
        #print("DBG: path:", audio_file_path, "loc-idx:", local_idx, "loc-range:", local_range)
        
        # Calculate start/end times and frames with corrections.
        # make sure the audio start time corresponds to a frame for which f0 was estimates with CREPE
        audio_start_seconds = local_idx * self.example_length / self.sample_rate + self.offset_correction
        audio_start_time = audio_start_seconds // 0.016 * 256 / self.sample_rate # seconds // crepe_hop_size [s]  * crepe_hop_size [samples] / sample_rate
        audio_length = self.example_length // 256 * 256 / self.sample_rate  # length in seconds
        crepe_start_frame = int(audio_start_time/0.016)
        crepe_end_frame = crepe_start_frame + int(audio_length / 0.016)
        if self.cunet_original:
            crepe_start_frame -= 2
            crepe_end_frame += 2
        
        name = audio_file_path.split('/')[-1].split('.')[0]
        
        # Load sources for the example.
        audio = utils.load_audio(audio_file_path, start=audio_start_time, dur=audio_length)
        # Channel indices are reversed since string 1 is highest and 6 lowest.
        channel_indices = np.array(self.allowed_strings[::-1]) - 1
        sources = audio[channel_indices].T # [n_samples, n_sources]

        if not self.f0_from_mix:
            # Load CREPE frequency
            confidence_file = os.path.join(self.crepe_dir, name + "_confidence.npy")
            confidences = np.load(confidence_file)[channel_indices, crepe_start_frame:crepe_end_frame]
            f0_file = os.path.join(self.crepe_dir, name + "_frequency.npy")
            frequency = np.load(f0_file)[channel_indices, crepe_start_frame:crepe_end_frame]
            frequency = np.where(confidences < self.conf_threshold, 0, frequency)
            frequencies = torch.from_numpy(frequency).type(torch.float32).T # [n_frames, n_sources]
        else:
            # Load cuesta frequency
            f0_from_mix_file = os.path.join(self.f0_cuesta_dir, + name + '.pt')
            f0_estimates = torch.load(f0_from_mix_file)[crepe_start_frame:crepe_end_frame, :]
            frequencies = f0_estimates

        # zero padding of last incomplete example frame in file
        n_zeros = self.example_length - sources.shape[0]
        if n_zeros != 0:
            zeros = torch.zeros(n_zeros, self.n_sources)
            sources = torch.cat((sources, zeros), dim=0)
            n_zeros = crepe_end_frame - crepe_start_frame - frequencies.shape[0]
            zeros = torch.zeros(n_zeros, self.n_sources)
            frequencies = torch.cat((frequencies, zeros), dim=0)

        if not self.cunet_original:
            assert sources.shape[0] // 256 == frequencies.shape[0], \
                'Sources and frequencies lengths are inconsistent!' + \
                f'Sources has frame length {sources.shape[0] / 256} and frequencies has length {frequencies.shape[0]}.'
        
        name += '_{}'.format(np.round(audio_start_time, decimals=3))
        
        # mix and normalize
        mix = torch.sum(sources, dim=1)  # [n_samples]
        mix_max = mix.abs().max()
        mix = mix / mix_max
        sources = sources / mix_max  # [n_samples, n_sources]
        
        if self.return_name: return mix, frequencies, sources, name, self.allowed_strings
        else: return mix, frequencies, sources



#------------------------------------------------------------------------------
# Tests

def plot_channels(channels, batch_num, allowed_strings, title=""):
    import matplotlib.pyplot as plt
    n_channels = channels.shape[2]
    plt.figure()
    fig, axs = plt.subplots(n_channels)
    plt.title(title)
    for i in range(n_channels):
        axs[i].plot(channels[batch_num, :, i].numpy())
        axs[i].set_ylabel("string" + str(allowed_strings[i]))
    plt.show()
    return fig

def testGuitarset():
    batch_size = 4
    # Testing function.
    ds = Guitarset(batch_size=batch_size,
                    dataset_range=(0, 8),
                    style='comp',
                    genres=['bn'],
                    allowed_strings=[1, 2, 3, 4, 5, 6],
                    shuffle_files=False)

    from torch.utils.data import DataLoader
    loader = iter(DataLoader(ds, batch_size=None, shuffle=False))
    mix, freqs, sources = next(loader)

    for batch in range(batch_size):    
        plot_channels(sources, batch, ds.allowed_strings, "Sources" + str(batch))
        
        # Plot freqs
        plot_channels(freqs, batch, ds.allowed_strings, "Freqs" + str(batch))

def debugEvalGuitarset():
    import models
    from torch.utils.data import DataLoader
    device = 'cpu'
    tag = 'guitarset_test'
    f0_cuesta = False
    
    trained_model, model_args = utils.load_model(tag, device, return_args=True)
    
    n_files_per_style_genre = model_args['n_files_per_style_genre']
    valid_split = model_args['valid_split']
    n_train_files = int((1 - valid_split) * n_files_per_style_genre)
    n_valid_files = int(valid_split * n_files_per_style_genre)
    # Use same amount of test data as validation data.
    # We use the next unused files of the Dataset.
    n_test_files = n_valid_files
    
    test_set = Guitarset(
        dataset_range=(n_train_files + n_valid_files,
                       n_train_files + n_valid_files + n_test_files),
        style=model_args['style'],
        genres=model_args['genres'],
        allowed_strings=model_args['strings'],
        shuffle_files=False,
        conf_threshold=model_args['confidence_threshold'],
        example_length=model_args['example_length'],
        return_name=True,
        f0_from_mix=f0_cuesta,
        cunet_original=False)
    
    # Ist das problem pbar statt iter(Dataloader(ds))?
    loader = iter(DataLoader(test_set, batch_size=1, shuffle=False))
    pbar = tqdm.tqdm(loader)
    
    #pbar = tqdm.tqdm(test_set)
    for d in pbar:
        print(d)

def testGuitarsetCompareFile():
    from torch.utils.data import DataLoader
    import scipy.io.wavfile as wavfile
    batch_size = 2
    ds = Guitarset(batch_size=batch_size,
                   dataset_range=(0, 4),
                   style='solo',
                   genres=['ss'],
                   allowed_strings=[1, 2, 3, 4, 5, 6],
                   shuffle_files=False)
    # Batching is done in the Guitarset, hence set batch_size=None in the DataLoader.
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=1)
    
    out_mix = []
    out_sources = []
    out_freqs = []
    for mix, freqs, sources in loader:
        out_mix.append(mix)
        out_sources.append(sources)
        out_freqs.append(freqs)
    
    out_mix = torch.cat(out_mix, dim=1)
    out_sources = torch.cat(out_sources, dim=1)
    out_sources = torch.transpose(out_sources, 1, 2)
    out_freqs = torch.cat(out_freqs, dim=1)
    #out_freqs = torch.transpose(out_freqs, 1, 2)
    
    for batch in range(batch_size):
        wavfile.write(f"{batch}_test_mix.wav", rate=16000, data=out_mix[batch].squeeze().numpy())
        wavfile.write(f"{batch}_test_sources.wav", rate=16000, data=out_sources[batch].squeeze().T.numpy())
        fig = plot_channels(out_freqs, batch, ds.allowed_strings, f"Batch {batch} sources f0")
        fig.savefig(f"{batch}_test_sources_f0.png", dpi=300)

if __name__ == "__main__":
    #debugEvalGuitarset()
    #testGuitarset()
    testGuitarsetCompareFile()

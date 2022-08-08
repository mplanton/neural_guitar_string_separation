# Unsupervised Audio Source Separation Using Differentiable Parametric Source Models

This is the source code for the experiments related to the paper [Unsupervised Audio Source Separation Using Differentiable Parametric Source Models](https://arxiv.org/abs/2201.09592).  

It contains a re-implementation of parts of the DDSP library in PyTorch. We added a differentiable all-pole filter which can be parameterized by line spectral frequencies or reflection coefficients. 

Please cite the paper, if you use parts of the code in your work.

**Modified by Manuel Planton for Guitar String separation.**


## Links
[:loud_sound: Audio examples](https://schufo.github.io/umss/)

[:page_facing_up: Paper](https://arxiv.org/abs/2201.09592)


## Requirements

The following packages are required:

    pytorch==1.6.0
    matplotlib==3.3.1
    python-sounddevice==0.4.0
    scipy==1.5.2
    torchaudio=0.6.0
    tqdm==4.49.0
    pysoundfile==0.10.3
    librosa==0.8.0
    scikit-learn==0.23.2
    tensorboard==2.3.0
    resampy==0.2.2
    pandas==1.2.3
    tensorboard==2.3.0


### Google Colab GPU environment dependencies

`!pip install torch==1.6.0 torchaudio==0.6.0 matplotlib sounddevice==0.4.0 scipy pysoundfile tqdm 
librosa==0.8.0 scikit-learn pandas resampy==0.2.2 tensorboard configargparse`


## Training

        python train.py -c config.txt
    
        python train_u_nets.py -c unet_config.txt

Resume training with:

        python train.py -c config.txt --wst-model path/to/checkpoint/directory

## Evaluation

        python eval.py --tag 'TAG' --f0-from-mix --test-set 'CSD' --no-baselines
        python eval.py --tag 'KarplusStrong_ex1' --test-set 'Guitarset'

### Show statistics of evaluation
        python eval_show_stats.py --tag 'TAG' --info-json 'path/to/info/TAG.json' --box-plot

## Inference

        python inference.py --tag 'TAG' --test-set CSD --song-name 'El Rossinyol'
        python inference.py --tag 'TAG' --test-set Guitarset --song-names 05_SS3-98-C_comp_hex_cln.wav 02_BN1-129-Eb_comp_hex_cln.wav

## Acknowledgment

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.

## Copyright

Copyright 2021 Kilian Schulze-Forster of Télécom Paris, Institut Polytechnique de Paris.
All rights reserved.

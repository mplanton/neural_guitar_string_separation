# Unsupervised Guitar String Music Source Separation

It contains a re-implementation of parts of the DDSP library in PyTorch.

[original code](https://github.com/schufo/umss)

[original paper](https://arxiv.org/abs/2201.09592)


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

## Examination

This bash script takes the trained model and

* plots the learning curves,
* does evaluation,
* plots evaluation statistics, and
* does inference.

        Usage: examination.sh <trained-model-path> <test-set> <which> <'list.wav of.wav test.wav fil    es.wav'>
        bash examination.sh trained_models/KarplusStrong_ex1/ Guitarset best '05_SS3-98-C_comp_hex_cln.wav 02_BN1-129-Eb_comp_hex_cln.wav'

(Do not forget the apostrophes around the list of test files.)
It does so by executing the following python scripts (plot_learning_curves.py, eval.py, eval_show_stats.py, inference.py, etc.).


### Evaluation

        python eval.py --tag 'KarplusStrong_ex1' --which best --test-set 'Guitarset'

### Show statistics of evaluation

        python eval_show_stats.py --tag 'TAG' --info-json 'path/to/info/TAG.json' --box-plot


### Compare models performances

        python eval_compare_models.py path/to/trained/model1 path/to/trained/model2 ...

The baseline metrics are taken from the first model path.
Plots are saved under `eval_compare/date/`.


### Inference

        python inference.py --tag 'TAG' --which best --test-set Guitarset --song-names 05_SS3-98-C_comp_hex_cln.wav 02_BN1-129-Eb_comp_hex_cln.wav


## Copyright

Copyright 2021 Kilian Schulze-Forster of Télécom Paris, Institut Polytechnique de Paris.
All rights reserved.

**Modified by Manuel Planton for Guitar String separation.**


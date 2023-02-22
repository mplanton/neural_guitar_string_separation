import pandas as pd
import json
import os
from pathlib import Path
import sys
import glob
import matplotlib.pyplot as plt
from datetime import datetime

# Usage:
# python eval_compare_models.py path/to/trained/model1 path/to/trained/model2 ...
#
# The baseline metrics are taken from the first model path.
# Plots are saved under eval_compare/date/.

# Expand the path list
model_paths_list = sys.argv[1:]
model_paths = [glob.glob(path) for path in model_paths_list]
# flatten the path list
model_paths = sorted([item for sublist in model_paths for item in sublist])

today = datetime.now()
save_path = os.path.join("eval_compare", today.strftime('%Y%m%d'))
Path(save_path).mkdir(parents=True, exist_ok=True)

metrics = ['sp_SNR', 'sp_SI-SNR', 'SI-SDR', 'mel_cep_dist']
evals = []

# Labels
labels = {
    "KarplusStrong_ex1" : "ExA1",
    "KarplusStrong_ex2" : "ExA2",
    "KarplusStrongC_ex1" : "ExC1",
    "KarplusStrongC_ex2" : "ExC2",
    "KarplusStrongB_ex0" : "ExB0",
    "KarplusStrongB_ex1" : "ExB1",
    "KarplusStrongB_ex2" : "ExB2",
    "KarplusStrongB_ex2_1" : "ExB2.1",
    "KarplusStrongB_ex3" : "ExB3",
    "KarplusStrongB_ex4" : "ExB4",
    "KarplusStrongC_ex1" : "ExC1",
    "KarplusStrongC_ex2" : "ExC2",
    
}

# Get the baseline metrics.
baseline_path = model_paths[0]

info_json = glob.glob(os.path.join(baseline_path, "*.json"))[0]
with open(info_json, 'r') as f:
    info = json.load(f)
baseline_args = info['args']
baseline_tag = baseline_args['tag']

# single voice f0 tracker used (monophonic pitch tracker)
f0_add_on = 'sf0'
# multi voice f0 tracker used (polyphonic pitch tracker)
if baseline_args['f0_cuesta']: f0_add_on = 'mf0'

ds = baseline_args['dataset']

irm_path = "evaluation/" + baseline_tag + "_IRM"
noise_path = "evaluation/" + baseline_tag + "_noise"

# IRM upper baseline
if os.path.exists(irm_path):
    eval_irm = pd.read_pickle(irm_path + "/eval_results_" + \
                                 f0_add_on + "_" + ds + "/all_results.pandas")
    eval_irm['name'] = "IRM"
    evals.append(eval_irm)

# Noise lower baseline
if os.path.exists(noise_path):
    eval_noise = pd.read_pickle(noise_path + "/eval_results_" + \
                                 f0_add_on + "_" + ds + "/all_results.pandas")
    eval_noise['name'] = "noise"
    evals.append(eval_noise)


# Get the metrics for direct synthesis and masking
for model_path in model_paths:
    info_json = glob.glob(os.path.join(model_path, "*.json"))[0]
    with open(info_json, 'r') as f:
        info = json.load(f)
    model_args = info['args']
    tag = model_args['tag']
    
    # single voice f0 tracker used (monophonic pitch tracker)
    f0_add_on = 'sf0'
    # multi voice f0 tracker used (polyphonic pitch tracker)
    if model_args['f0_cuesta']: f0_add_on = 'mf0'
    
    ds = model_args['dataset']

    # Direct synthesis
    eval_synth = pd.read_pickle("evaluation/" + tag + "/eval_results_" + \
                                 f0_add_on + "_" + ds + "/all_results.pandas")
    if tag not in labels.keys():
        eval_synth['name'] = tag + "_synth"
    else:
        eval_synth['name'] = labels[tag] + "_synth"
    evals.append(eval_synth)
    
    # Masked
    eval_masked = pd.read_pickle("evaluation/" + tag + "_masking/eval_results_" + \
                                 f0_add_on + "_" + ds + "/all_results.pandas")
    if tag not in labels.keys():
        eval_masked['name'] = tag + "_masked"
    else:
        eval_masked['name'] = labels[tag] + "_masked"
    evals.append(eval_masked)


ylabels = {
    'sp_SNR': 'sp. SNR [dB]',
    'sp_SI-SNR': 'sp. SI-SNR [dB]',
    'SI-SDR': 'SI-SDR [dB]',
    'mel_cep_dist': 'Mel Cepstral Dist.'
}

for metric in metrics:
    # Accumulate metrics
    df_metric = {}
    for ev in evals:
        df_metric[ev['name'][0]] = ev[metric]
    df_metric = pd.DataFrame(df_metric)
    
    # Metrics boxplots
    fig = plt.figure(figsize=[12, 9])
    df_metric.boxplot()
    plt.ylabel(ylabels[metric])
    plt.xticks(rotation=90)
    plt.tight_layout()
    f_path = os.path.join(save_path, "compare_" + metric + ".pdf")
    fig.savefig(f_path, dpi=300)
    
    # Median Ranking
    fig = plt.figure(figsize=[12, 9])
    ax = df_metric.median().sort_values().plot.bar()
    ax.set_ylabel("median " + ylabels[metric])
    plt.tight_layout()
    f_path = os.path.join(save_path, "median_ranking_" + metric + ".pdf")
    fig.savefig(f_path, dpi=300)
    
    # Mean Ranking
    fig = plt.figure(figsize=[12, 9])
    ax = df_metric.mean().sort_values().plot.bar()
    ax.set_ylabel("mean " + ylabels[metric])
    plt.tight_layout()
    f_path = os.path.join(save_path, "mean_ranking_" + metric + ".pdf")
    fig.savefig(f_path, dpi=300)

print("Results saved in", save_path)

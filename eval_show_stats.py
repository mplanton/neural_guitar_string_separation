import pandas as pd
import argparse
import json
import os
import matplotlib.pyplot as plt

# python eval_show_stats.py --tag 'TAG' --info-json 'path/to/info/TAG.json'

parser = parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, help="Model tag.")
parser.add_argument('--info-json', type=str, help="Path to the JSON file containing the model info like args and training information.")
parser.add_argument('--box-plot', action='store_true', default=False, help=\
                    "Plot the metrics as box plots.")

args, _ = parser.parse_known_args()

with open(args.info_json, 'r') as f:
    info = json.load(f)
model_args = info['args']

# single voice f0 tracker used (monophonic pitch tracker)
f0_add_on = 'sf0'
# multi voice f0 tracker used (polyphonic pitch tracker)
if model_args['f0_cuesta']: f0_add_on = 'mf0'

ds = model_args['dataset']
metrics = ['sp_SNR', 'sp_SI-SNR', 'SI-SDR', 'mel_cep_dist']


# Direct synthesis
eval_synth = pd.read_pickle("evaluation/" + args.tag + "/eval_results_" + \
                             f0_add_on + "_" + ds + "/all_results.pandas")
eval_synth['name'] = "synth"
print("Direct synthesis stats:")
print(eval_synth[metrics].describe())

# Masked
eval_masked = pd.read_pickle("evaluation/" + args.tag + "_masking/eval_results_" + \
                             f0_add_on + "_" + ds + "/all_results.pandas")
eval_masked['name'] = "masked"
print("\n", 80*'-', "\n")
print("Masked mix stats:")
print(eval_masked[metrics].describe())


evals = [eval_synth, eval_masked]

# Baselines
irm_path = "evaluation/" + args.tag + "_IRM"
noise_path = "evaluation/" + args.tag + "_noise"

# IRM upper baseline
if os.path.exists(irm_path):
    eval_irm = pd.read_pickle(irm_path + "/eval_results_" + \
                                 f0_add_on + "_" + ds + "/all_results.pandas")
    eval_irm['name'] = "IRM"
    evals.append(eval_irm)
    print("\n", 80*'-', "\n")

    print("IRM upper baseline stats:")
    print(eval_irm[metrics].describe())

# Noise lower baseline
if os.path.exists(noise_path):
    eval_noise = pd.read_pickle(noise_path + "/eval_results_" + \
                                 f0_add_on + "_" + ds + "/all_results.pandas")
    eval_noise['name'] = "noise"
    evals.append(eval_noise)
    print("\n", 80*'-', "\n")

    print("White noise as lower baseline stats:")
    print(eval_noise[metrics].describe())

# Box plot
if args.box_plot:
    for metric in metrics:
        # Accumulate metrics
        df_metric = {}
        for ev in evals:
            df_metric[ev['name'][0]] = ev[metric]
        # Plot
        df_metric = pd.DataFrame(df_metric)
        fig = plt.figure()
        df_metric.boxplot()
        plt.title(metric)
        fig.savefig(args.tag + "_" + metric + ".png", dpi=300)
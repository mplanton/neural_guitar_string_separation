#!/bin/bash

# Usage: examination.sh <trained-model-path> <test-set> <which> <'list.wav of.wav test.wav files.wav'>
#
# Your conda environment must be active when executing this script!

# examination.sh
#
# This script takes a trained model and
# * plots learning curves
# * does evaluation
# * plots evaluation statistics
# * does inference
# * plots inferred data

printUsage () {
    echo "Usage: examination.sh <trained-model-path> <test-set> <which> <'list.wav of.wav test.wav files.wav'>"
    echo
    echo "Your conda environment must be active when executing this script!"
    echo
    echo This examination script
    echo * plots the learning curves,
    echo * does evaluation,
    echo * plots evaluation statistics, and
    echo * does inference.
    echo
    echo "<trained-model-path>: The path of the directory, the trained model is stored."
    echo "    The config*.txt and JSON file have to be in this directory."
    echo
    echo "<test-set>: Name of the test set to use for evaluation and inference (e.g. Guitarset)."
    echo
    echo "<which>: Which model to load (best or last). The best model on validation or the last trained one."
    echo
    echo "<list-of-test-files>: The list of test files are used for inference (Give at least batch_size test files for inference to work)."
}

# Check if all three arguments are given.
if [ ${#1} -eq 0 ] || [ ${#2} -eq 0 ] || [ ${#3} -eq 0 ] || [ ${#4} -eq 0 ]
then
    echo Missing arguments!
    echo
    printUsage
    exit -2
fi

# Check for requesting help.
if [ "$1" = "-h" ] || [ "$2" = "-h" ] || [ "$3" = "-h" ] || \
   [ "$1" = "--help" ] || [ "$2" = "--help" ] || [ "$3" = "--help" ]
then
    printUsage
fi

trained_model_path=$1
test_set=$2
which_model=$3
test_files=$4

config_file="config*.txt"

# Check for valid trained model path.
len=${#trained_model_path}
if [ $len -eq 0 ]
then
    echo "The trained model path is empty!"
    printUsage
    exit -1
elif [ ! -d $trained_model_path ]
then
    echo "The trained model path $trained_model_path does not exist!"
    exit -1
fi

# Check for valid config file.
n_conf_files=$(ls 2>/dev/null -Ubad1 -- $trained_model_path$config_file | wc -l)
if [ $n_conf_files -gt 1 ]
then
    echo "The trained model path $trained_model_path has more than one config*.txt file!"
    exit -1
elif [ $n_conf_files -eq 0 ]
then
    echo "The trained model path $trained_model_path has no config*.txt file!"
    exit -1
fi


# Get the model tag from the config file
tag=$(cat $trained_model_path$config_file | grep -i "tag:" | awk '{print $2}')

# Create plots directory
examination_path="examination/"
plots_path=$examination_path$tag"_plots/"
mkdir -p $plots_path


json_file=$trained_model_path$tag".json"

echo "Plot learning curves."
python3 plot_learning_curves.py $json_file
f_names="_learning_curves.*"
mv $tag$f_names $plots_path
echo "Saved learning curves to $plots_path."
echo

echo
echo "Inference..."
python3 inference.py --tag $tag --which $which_model --test-set $test_set --song-names $test_files
echo "Inference done."

echo "Evaluation..."
python3 eval.py --tag $tag --which $which_model --test-set $test_set
echo "Evaluation done."

echo "Plot evaluation statistics."
python3 eval_show_stats.py --tag $tag --info-json $json_file --box-plot
f_names="_eval_*.*"
mv $tag$f_names $plots_path
echo "Saved evaluation statistics plots to $plots_path."

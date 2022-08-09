import sys
import matplotlib.pyplot as plt
import json

# Usage: plot_learning_curves.py <path_to_json_file.json>
#
# Plots the learning curves of the given model.


with open(sys.argv[1], "r") as f:
    data = json.load(f)
    train_loss = data['train_loss_history']
    valid_loss = data['valid_loss_history']
    
    fig = plt.figure()
    plt.plot(train_loss, label="train loss")
    plt.plot(valid_loss, label="validation loss")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    #plt.title("Karplus-Strong model trained with Guitarset")
    fig.savefig(sys.argv[1].split('/')[-1].split('.')[0] + "_learning_curves.png", dpi=300)

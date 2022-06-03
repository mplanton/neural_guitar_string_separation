#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script converts the CSV files output from CREPE to:
  - FILENAME_confidence.npy
  - FILENAME_frequency.npy
"""
import glob
import os
import numpy as np

csd_path = os.path.abspath("../Datasets/ChoralSingingDataset")
songs_paths = glob.glob(os.path.join(csd_path, "*"))

for song_path in songs_paths:
    print("DBG:", song_path)
    
    crepe_path = os.path.join(song_path, "crepe_f0_center")
    csv_paths = glob.glob(os.path.join(crepe_path, "*.csv"))
    for csv_path in csv_paths:
        print("  DBG:", csv_path)
        
        # Get column names.
        column_names = []
        with open(csv_path, 'r') as csv_file:
            line = csv_file.readline()[:-1] # slice off \n
            column_names = line.split(',')
            print("--DBG:", column_names)
        
        # Get data without column names.
        csv_data = np.genfromtxt(csv_path, delimiter=',')[1:]
        print("    DGB:", csv_data)
        
        # Save the columns separately in npy files
        for i, column_name in enumerate(column_names):
            data = csv_data[:, i]
            npy_path = csv_path.split('.')[0] + "_" + column_name + ".npy"
            with open(npy_path, 'wb') as f:
                np.save(f, data)
import argparse
import numpy as np
import math
import glob
from random import randint

parser = argparse.ArgumentParser()
parser.add_argument("n_tces", help="Number of individual TCEs", type=int)
parser.add_argument("prop_val", help="Proportion to be shifted to validation", type=float)
parser.add_argument("input_set", help="Path to training set directory")
args = parser.parser_args()

target_amount = math.floor(args.n_tces * args.prop_val)
set_dir = args.input_set

set_dir = "toi_match_7_cross_val_set/"
train_path = set_dir+"train_lcs/"
val_path = set_dir+"val_lcs/"


def fetchPairedPath(path):
    last_section_arr = (path.split("/")[-1]).split("-")
    tic_id = last_section_arr[0]
    tce_id = last_section_arr[1]
    both_paths = glob.glob(train_path+tic_id+"-"+tce_id+"*.npy")
    return both_paths

def trainToValPath(path):
    last_section = path.split("/")[-1]
    return val_path+last_section

def shiftFiles(both_paths):
    for i in range(0, 2):
        path = both_paths[i]
        temp = np.load(path)
        np.save(trainToValPath(path), temp)

file_paths = glob.glob(train_path+"*local.npy")
used_indices = []

random_index = randint(0, total_files-1)
local_path = file_paths[random_index]
both_paths = fetchPairedPath(local_path)
shiftFiles(both_paths)

used_indices.append(random_index)
count_of_total_shifted = 1
while (count_of_total_shifted < target_amount):
    random_index = randint(0, total_files-1)
    while (random_index in used_indices):
        random_index = randint(0, total_files-1)
    local_path = file_paths[random_index]
    both_paths = fetchPairedPath(local_path)
    shiftFiles(both_paths)
    used_indices.append(random_index)
    count_of_total_shifted += 1
    print(f"Processed ${count_of_total_shifted}")

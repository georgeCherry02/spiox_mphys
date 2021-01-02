import argparse
import math

import numpy as np
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("tce_list", help="File path of tce list")
parser.add_argument("toi_list", help="File path of toi list")
parser.add_argument("exceptions_list", help="List of TOI exceptions due to rounding errors")
parser.add_argument("old_list", help="File path of file to correct")
args = parser.parse_args()

# Load in files
toi_list = pd.read_csv(args.toi_list, skiprows=4)
tce_list = pd.read_csv(args.tce_list, skiprows=6)
exceptions_list = pd.read_csv(args.exceptions_list)
to_correct_list = pd.read_csv(args.old_list)

def normalRound(val):
    return math.floor(val) if ((val - math.floor(val)) < 0.5) else math.ceil(val)

def compareValues(tce_val, toi_val, tce_id):
    # Determine degree of precision of tce_val
    if (np.isnan(tce_val) or np.isnan(toi_val)):
        return False
    dp = 0
    while (tce_val % 1 != 0):
        dp += 1
        # Quick attempt at fixing floating point accuracy problem
        tce_val = round(tce_val, 8)
        tce_val *= 10
    toi_val = toi_val * pow(10, dp)
    return tce_val == normalRound(toi_val)

# Define TCE comparison function
# More precise values are recorded in the TOI file
# This list is definitely not an exhaustive list of properties but I believe it to be sufficient
key_pairs = [["transitEpochBtjd", "Epoch Value", "e"], ["transitDepthPpm", "Transit Depth Value", "de"], ["transitDurationHours", "Transit Duration Value", "du"], ["orbitalPeriodDays", "Orbital Period Value", "o"]]
def matchTCE(tce_data, toi_data):
    tce_id = tce_data["tceid"]
    factor_count = 0
    map_test = {}

    for [tceKey, toiKey, mapKey] in key_pairs:
        tce_val = tce_data[tceKey].values[0]
        toi_val = toi_data[toiKey].values[0]
        similar = compareValues(tce_val, toi_val, tce_id)
        map_test[mapKey] = similar
        if similar:
            factor_count += 1
    
    return factor_count == 4

# Begin to loop through all the tces in the training set csv to correct
for index, training_set_elem in to_correct_list:
    tce_id = training_set_elem["tce_id"]
    tic_id = training_set_elem["tic_id"]
    toi_data = toi_list[toi_lis["TIC"] == tic_id]
    tce_data = tce_list[tce_list["tce_id"] == tce_id]
    print(f"Processing TIC {tic_id}")
    if (tce_id in exceptions_list["ids"]):
        to_correct_list.at(index, "pc") = 1
    elif (len(toi_data) > 0):
        to_correct_list.at(index, "pc") = 1 if (matchTCE(tce_data, toi_data)) else 0
    else:
        to_correct_list.at(index, "pc") = 0

to_correct_list.to_csv(args.old_list)

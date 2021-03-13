import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("toi_path", help="Directory to get tois from")
parser.add_argument("number_shared", help="Number of shared features within givin accuracy")
parser.add_argument("accuracy", help="Match accuracy, i.e. features need to be within 10% if this is 0.1")
args = parser.parse_args()

# Define comparison functions
def compareValues(tce_val, toi_val):
    ratio = toi_val/tce_val
    return (ratio <= (1 + float(args.accuracy)) and ratio >= (1 - float(args.accuracy)))

key_pairs = [["transitEpochBtjd", "Epoch Value"], ["transitDepthPpm", "Transit Depth Value"], ["transitDurationHours", "Transit Duration Value"], ["orbitalPeriodDays", "Orbital Period Value"]]
def matchTCE(tce_entry, toi_entry):
    similarity_count = 0
    for [tceKey, toiKey] in key_pairs:
        tce_val = tce_entry[tceKey].values[0]
        toi_val = toi_entry[toiKey]
        if compareValues(tce_val, toi_val):
            similarity_count += 1
    return similarity_count

# Begin loop through previously processed tces
def correctSector(col_data, tce_data, toi_data):
    init_sum = col_data["pc"].sum()
    print(f"Init sum: {init_sum}")
    for ptce_index, processed_tce in col_data.iterrows():
        tic_id = processed_tce["tic_id"]
        tce_id = processed_tce["tce_id"]
        tce_entry = tce_data[tce_data["tceid"]==tce_id]
        toi_entries = toi_data[toi_data["TIC"]==tic_id]
        # Now loop through found TOIs and see which matches TCE
        match_found = False
        if (toi_entries.size > 0):
            for toi_index, toi_entry in toi_entries.iterrows():
                status = matchTCE(tce_entry, toi_entry)
                if (status >= int(args.number_shared)):
                    match_found = True
                    break
        col_data.at[ptce_index, "pc"] = 1 if match_found else 0
    fin_sum = col_data["pc"].sum()
    print(f"Fin sum: {fin_sum}")
    return col_data

toi_data = pd.read_csv(args.toi_path, skiprows=4)
for sector in range(5, 26):
    if (sector == 12):
        continue
    print(f"Processing sector {sector}")
    # Read in necessary files for each sector
    collated_path = "s"+str(sector)+"_training_set/collated_scientific_domain_parameters.csv"
    col_data = pd.read_csv(collated_path)
    tce_path = "tce_list_s"+str(sector)+".csv"
    tce_data = pd.read_csv(tce_path, skiprows=6)
    # Correct the collated data
    col_data = correctSector(col_data, tce_data, toi_data)
    # Overwrite collated data
    col_data.to_csv(collated_path)

print("Process complete")

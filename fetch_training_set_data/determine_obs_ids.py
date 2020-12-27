### Imports
import argparse
import pandas as pd

from astroquery.mast import Observations

### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("in_file", help="Path to csv file containing TCEs for sector")
parser.add_argument("out_file", help = "File to write tic id list output to")
args = parser.parse_args()

### Read in the tce data
tce_data = pd.read_csv(args.in_file, skiprows=6)
tic_ids = tce_data["ticid"]

# This is because there are multiple TCEs for individual tic ids
tic_ids_already_checked = []
obs_ids_with_two_min_cadences = []

for tic_id in tic_ids:
    if tic_id in tic_ids_already_checked:
        continue
    tic_ids_already_checked.append(tic_id)

    tic_id = "TIC "+str(tic_id)
    print(f"Looking for two min observations of {tic_id}")
    two_min_observations = Observations.query_criteria(objectname=tic_id, dataproduct_type="timeseries", obs_collection="TESS", obs_id="*-0120-s")
    if (len(two_min_observations) > 0):
        print(f"Found observations fitting description for {tic_id}")
        for obs_id in two_min_observations["obsid"]:
            obs_ids_with_two_min_cadences.append(obs_id)

output = "obs_id\n"
for obs_id in obs_ids_with_two_min_cadences:
    output += str(obs_id)+"\n"

with open(args.out_file, 'w') as f:
    f.write(output)

print("Process complete")

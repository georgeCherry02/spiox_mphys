### Imports
import argparse
import pandas as pd

from astroquery.mast import Observations

### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("in_file", help="Path to csv file containing TCEs for sector")
parser.add_argument("out_file", help="File to write tic id list output to")
parser.add_argument("sector", help="Sector number observations are from")
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

    only_tic_id = str(tic_id)
    tic_id = "TIC "+only_tic_id
    print(f"Looking for two min observations of {tic_id}")
    obs_name_pattern = "*-s*"+args.sector+"-0000*"+only_tic_id+"*-s"
    two_min_observations = Observations.query_criteria(objectname=tic_id, dataproduct_type="timeseries", obs_collection="TESS", obs_id=obs_name_pattern)
    if (len(two_min_observations) > 0):
        print(f"Found observations fitting description for {tic_id}")
        ### For some reason this returns multiple tic id observations?
        ### Therefore filter by target_name again
        two_min_observations = two_min_observations[two_min_observations["target_name"] == only_tic_id]
        for obs_id in two_min_observations["obsid"]:
            obs_ids_with_two_min_cadences.append(obs_id)

output = "obs_id\n"
for obs_id in obs_ids_with_two_min_cadences:
    output += str(obs_id)+"\n"

with open(args.out_file, 'w') as f:
    f.write(output)

print("Process complete")

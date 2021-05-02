### Imports
import argparse
import pandas as pd

from astroquery.mast import Observations

### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("in_file", help="Path to csv file containing TCEs for sector")
parser.add_argument("out_file", help="File to write obs id list output to")
parser.add_argument("sector", help="Sector number observations are from")
args = parser.parse_args()

### Read in the tce data
tce_data = pd.read_csv(args.in_file, skiprows=6)
tic_ids = tce_data["ticid"]

# This is because there are multiple TCEs for individual tic ids
tic_ids_already_checked = []
obs_ids_with_two_min_cadences = []

last_logged_index = 0

with open(args.out_file, 'w') as f:
    f.write("obs_id\n")

for tic_id in tic_ids:
    if tic_id in tic_ids_already_checked:
        continue
    tic_ids_already_checked.append(tic_id)

    only_tic_id = str(tic_id)
    tic_id = "TIC "+only_tic_id
    print(f"Looking for two min observations of {tic_id}")
    obs_name_pattern = "*-s*"+args.sector+"-*"+only_tic_id+"*-s"
    two_min_observations = Observations.query_criteria(objectname=tic_id, dataproduct_type="timeseries", obs_collection="TESS", obs_id=obs_name_pattern)
    if (len(two_min_observations) > 0):
        print(f"Found observations fitting description for {tic_id}")
        ### For some reason this returns multiple tic id observations?
        ### Therefore filter by target_name again
        two_min_observations = two_min_observations[two_min_observations["target_name"] == only_tic_id]
        for obs_id in two_min_observations["obsid"]:
            obs_ids_with_two_min_cadences.append(obs_id)

    if (len(obs_ids_with_two_min_cadences) != last_logged_index and len(obs_ids_with_two_min_cadences) % 20 == 0):
        print("Logging!")
        output = ""
        for i in range(0, 20):
            index = len(obs_ids_with_two_min_cadences) - i - 1
            output += str(obs_ids_with_two_min_cadences[index])+"\n"
        with open(args.out_file, 'a') as f:
            f.write(output)
        last_logged_index = len(obs_ids_with_two_min_cadences)

final_output = ""
for i in range(last_logged_index, len(obs_ids_with_two_min_cadences)):
    final_output += obs_ids_with_two_min_cadences[i]+"\n"
with open(args.out_file, 'a') as f:
    f.write(final_output)

total_length = len(tic_ids_already_checked)
found_length = len(obs_ids_with_two_min_cadences)
print(f"Proportion found {found_length}/{total_length}")
print("Process complete")

### Imports
import argparse
import os
import pandas as pd

from astroquery.mast import Observations

### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("in_file", help="Path to csv file containing TCEs for sector")
parser.add_argument("out_dir", help="Directory to put mastDownload file in")
args = parser.parse_args()


### Read in tce data
tce_data = pd.read_csv(args.in_file, skiprows=6)
tic_ids = tce_data["ticid"]

### Change to correct directory
os.chdir(args.out_dir)

count = 0
for tic_id in tic_ids:
    count += 1
    if (count > 3):
        break
    tic_id = "TIC "+str(tic_id)
    print("Looking for observations of "+tic_id)
    observations = Observations.query_criteria(objectname=tic_id, dataproduct_type="timeseries", obs_collection="TESS", obs_id="*-0120-s")
    print(f'Found {len(observations)} observations matching the criteria')
    data_products = Observations.get_product_list(observations["obsid"])
    manifest = Observations.download_products(data_products, description="Data validation time series")
    print(manifest)

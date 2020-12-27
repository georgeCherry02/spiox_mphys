### Imports
import argparse
import math
import os
import pandas as pd

from astroquery.mast import Observations

### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("in_file", help="CSV file containing all obs ids to download dv files from")
parser.add_argument("out_dir", help="Directory to output downloaded files to")
args = parser.parse_args()

### Read in tic ids to request dataproducts from
obs_data = pd.read_csv(args.in_file)
obs_ids = obs_data["obs_id"]
obs_ids = obs_ids.astype("str")

### Change to output directory
os.chdir(args.out_dir)

### Split into batches
for i in range(0, math.ceil(len(obs_ids)/300)):
    print(f"Doing batch {(i+1)}")
    upper_limit = min(len(obs_ids), (i+1) * 300)
    batch = obs_ids.take(range(i * 300, upper_limit))
    ### Fetch appropriate dataproducts
    data_products = Observations.get_product_list(batch)
    ### Download dataproducts
    manifest = Observations.download_products(data_products, description="Data validation time series")

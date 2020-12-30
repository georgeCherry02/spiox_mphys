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
parser.add_argument("n_batch", help="Number to download at once", type=int)
args = parser.parse_args()

### Read in tic ids to request dataproducts from
obs_data = pd.read_csv(args.in_file)
obs_ids = obs_data["obs_id"]
obs_ids = obs_ids.astype("str")

### Change to output directory
os.chdir(args.out_dir)

### Split into batches
for i in range(0, math.ceil(len(obs_ids)/args.n_batch)):
    print(f"Doing batch {(i+1)}")
    upper_limit = min(len(obs_ids), (i+1) * args.n_batch)
    batch = obs_ids.take(range(i * args.n_batch, upper_limit))
    ### Fetch appropriate dataproducts
    data_products = Observations.get_product_list(batch)
    ### Download dataproducts
    manifest = Observations.download_products(data_products, description="Data validation time series")

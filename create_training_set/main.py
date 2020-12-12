import argparse
import glob

import api_queries
import plot_data
import process_data
from file_handler import FileHandler

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", help="Directory to take raw data from")
parser.add_argument("out_dir", help="Directory to output np files to")
parser.add_argument("tce_data_dir", help="Directory to find tce csv data from")
args = parser.parse_args()

pattern = args.in_dir+"*-0120-s/*_dvt.fits"
fh = FileHandler(args.out_dir, args.in_dir, args.tce_data_dir)
# write_data.initialise_collated_parameters_file(args.out_dir)

files = glob.glob(pattern)
for file_path in files:
    # Parse file path for useful information
    [tic_id, tce_id, sector] = fh.parseDVFilePath(file_path)
    # Read in raw data
    dv_hdul = fh.readInDVFile(file_path)
    centroid_data = fh.readInCentroidData(tic_id, sector)
    dv_data = dv_hdul[1].data
    centroid_data = fh.readInCentroidData(tic_id, sector)
    tce_data = fh.readInTCEData(tce_id, sector)
    # Gather key variables for data processing
    epoch = float(tce_data["transitEpochBtjd"])
    period = float(tce_data["orbitalPeriodDays"])
    duration_hrs = float(tce_data["transitDurationHours"])
    duration = duration_hrs / 24
    # Normalise and quality correct required data
    [lc, centroid, time] = process_data.normaliseAndQualityCorrectData(dv_data, centroid_data, epoch)
    phase_folded_time = (time + period/2) % period
    binned_lc = process_data.binSeries(period, duration, lc, phase_folded_time)
    binned_cent = process_data.binSeries(period, duration, centroid, phase_folded_time)
    # Plot data
    plot_data.output(binned_lc, binned_cent, args.out_dir)
    # Get event parameters
    tic_data = api_queries.fetchTICEntry(tic_id)[0]
    headers = dv_hdul[0].header
    event_parameters = process_data.collateParameters(tce_id, tce_data, tic_data, headers, period, duration)
    # Write all information to appropriate files
    fh.appendParameters(event_parameters)
    fh.writeProcessedCurves(tce_id, lc, centroid, phase_folded_time)

print("Finished")

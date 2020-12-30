import argparse
import glob

import api_queries
import plot_data
import process_data
from file_handler import FileHandler

### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("dv_in_dir", help="Directory to take data validation fits from")
parser.add_argument("lc_in_dir", help="Directory to take raw lc fits from")
parser.add_argument("out_dir", help="Directory to output np files to")
parser.add_argument("tce_data", help="Path to tce data csv")
parser.add_argument("toi_data", help="Path to toi data csv")
parser.add_argument("sector", help="Sector number")
args = parser.parse_args()

### Find files to parse
fh = FileHandler(args.dv_in_dir, args.lc_in_dir, args.out_dir, args.tce_data, args.toi_data)
tce_data = fh.readInTCEData()
toi_data = fh.readInTOIData()

count = 0

for index, tce in tce_data.iterrows():
    if (count >= 3):
        break
    tic_id = tce["ticid"]
    tce_id = tce["tceid"]
    # Create file pattern
    [dv_hdul, lc_hdul] = fh.loadRawData(tic_id, args.sector)
    if (not lc_hdul):
        print(f'No lc or dv found, skipping tce: {tce["tceid"]}')
        continue
    count += 1
    raw_lc_data = lc_hdul[1].data
    dv_data = dv_hdul[int(tce["planetNumber"])].data
    dv_headers = dv_hdul[0].header
    # Gather key variables for data processing
    epoch = float(tce["transitEpochBtjd"])
    period = float(tce["orbitalPeriodDays"])
    duration_hrs = float(tce["transitDurationHours"])
    duration = duration_hrs / 24
    # Normalise and quality correct required data
    [lc, centroid, time] = process_data.normaliseAndQualityCorrectData(dv_data, raw_lc_data, epoch)
    # Bin the data
    phase_folded_time = (time + period/2) % period
    binned_lc = process_data.binSeries(period, duration, lc, phase_folded_time)
    binned_cent = process_data.binSeries(period, duration, centroid, phase_folded_time)
    # Plot the data
    plot_data.output(tce_id, binned_lc, binned_cent, args.out_dir)
    
    # Get event parameters
    tce_represents_pc = process_data.determineCandidateStatus(tic_id, toi_data)
    tic_data = api_queries.fetchTICEntry(str(tic_id))[0]
    event_parameters = process_data.collateParameters(tce_id, tce, tic_data, dv_headers, period, duration, tce_represents_pc)
    # Write all the information to appropriate files
    fh.appendParameters(event_parameters)
    fh.writeProcessedCurves(tce_id, binned_lc, binned_cent)

print("Finished")

import argparse
import glob
import warnings

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
parser.add_argument("--skipping", help="Skip until you find an unprocessed tce", action="store_true")
args = parser.parse_args()

# Numpy doesn't like Nans much and that's fair enough
warnings.filterwarnings("ignore")

def progressBar(count, set_size):
    output = "["
    for i in range(20, set_size, 20):
        if (i < count):
            output += "|"
        else:
            output += " "
    output += "]"
    print(output)

### Find files to parse
fh = FileHandler(args.dv_in_dir, args.lc_in_dir, args.out_dir, args.tce_data, args.toi_data, args.skipping)
tce_data = fh.readInTCEData()
toi_data = fh.readInTOIData()

### Define console output values
tce_amount = len(tce_data.index)
count = 0
percentage = round((count / tce_amount) * 100, 2)

### Define TICs to skip
odd_tics = {
    '9': [140055734, 179304342, 35975843, 38603673, 386072709, 388684102, 342829812],
    '10': [307027599, 307440881, 377174003, 432829812, 464402195],
    '11': [179304342, 241683370, 241728550, 253097689, 300104749, 314899629, 316295827, 398363472, 399977856, 415937547, 419944060, 441103771, 471015273],
    '12': [],
    '13': [],
    '14': [64799884, 68369092, 86319584],
    '15': []
}

tces_that_failed_to_normalise = []

for index, tce in tce_data.iterrows():
    tic_id = tce["ticid"]
    tce_id = tce["tceid"]
    if args.skipping and fh.decideSkip(tce_id):
        print("Skipping")
        count += 1
        progressBar(count, tce_amount)
        continue
    print(f"Processing TCE: {tce_id}")
    # Create file pattern
    [dv_hdul, lc_hdul] = fh.loadRawData(tic_id, args.sector)
    if (not lc_hdul):
        count += 1
        print(f'No lc or dv found, skipping tce: {tce["tceid"]}')
        fh.appendSkippedTCE(tce_id)
        continue
    raw_lc_data = lc_hdul[1].data
    tce_number = int(tce["planetNumber"])
    if (tce_number >= len(dv_hdul)):
        print("Odd trigger!")
        tce_number = 1
    dv_data = dv_hdul[tce_number].data
    dv_headers = dv_hdul[0].header
    # Gather key variables for data processing
    epoch = float(tce["transitEpochBtjd"])
    period = float(tce["orbitalPeriodDays"])
    duration_hrs = float(tce["transitDurationHours"])
    duration = duration_hrs / 24
    detected_depth = float(tce["transitDepthPpm"])
    # First phase fold and median bin both time series
    [binned_series, phase_folded_time] = process_data.prepareBinnedTimeSeries(period, duration, epoch, raw_lc_data, dv_data)
    binned_series = process_data.normaliseBinnedTimeSeries(duration, period, detected_depth, binned_series)
    binned_series = process_data.correctBinnedNans(duration, period, binned_series)
    [normalised_series, valid_normalisation] = process_data.finalNormalisation(binned_series)
    if (not valid_normalisation):
        print("Failed to do final normalisation")
        tces_that_failed_to_normalise.append(tce_id)
        count += 1
        progressBar(count, tce_amount)
        continue

    # Plot the data
    ### Comment out to avoid plotting
    # plot_data.output(tce_id, binned_series["lc"], binned_series["cent"], args.out_dir)
    
    # Get event parameters
    tce_represents_pc = process_data.determineCandidateStatus(tce, toi_data)
    event_parameters = process_data.collateParameters(tce_id, tce, dv_headers, period, duration, tce_represents_pc)
    # Write all the information to appropriate files
    fh.appendParameters(event_parameters)
    fh.writeProcessedCurves(tce_id, binned_series["lc"], binned_series["cent"])
    count += 1
    percentage = round((count / tce_amount) * 100, 2)
    print(f"{percentage}% Complete")
    progressBar(count, tce_amount)

print("Finished")


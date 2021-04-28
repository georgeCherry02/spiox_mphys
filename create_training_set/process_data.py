import math
import numpy as np
from scipy import constants as scientific_constants

### Old normalisation function
def normaliseSeries(series):
    med = np.median(series)
    series = series / med - 1
    sd = np.std(series)
    series /= sd
    return series

### Old data extraction
def normaliseAndQualityCorrectData(dv_data, centroid_data, epoch):
    time = dv_data["TIME"]
    if ("LC_DETREND" in dv_data.keys()):
        lc_detrend = dv_data["LC_DETREND"]
    else:
        lc_detrend = dv_data["LC_INIT"]
    # Clean data from dv file
    lc_nans = np.isnan(lc_detrend)
    lc_detrend = lc_detrend[np.logical_not(lc_nans)]
    time = time[np.logical_not(lc_nans)]
    # Clean data from lc file
    cent_x = centroid_data["MOM_CENTR1"]
    cent_y = centroid_data["MOM_CENTR2"]
    cent_nans = np.isnan(cent_x)
    cent_x = cent_x[np.logical_not(cent_nans)]
    cent_y = cent_y[np.logical_not(cent_nans)]
    # Shift epochs
    time -= epoch
    # Extract centroid
    rad = []
    for i in range(0, len(cent_x)):
        rad.append(math.sqrt(cent_x[i]**2 + cent_y[i]**2))
    # Normalise series
    lc_detrend = normaliseSeries(lc_detrend)
    rad = normaliseSeries(rad)
    return [lc_detrend, rad, time]

# Define constants for binning
n_global_bins = 2001
n_local_bins = 201
local_observation_width = 4
local_bin_width = 0.16

### Old binning functionality
def addToSeriesBins(bins, index, value):
    while (len(bins) <= index):
        bins.append([])
    bins[index].append(value)
    return bins

# Still used
def determineContainingBins(center, center_spacing, bin_width, begin, end, time):
    result = [center]
    found_top = False
    found_bottom = False
    j = 1
    while (not found_top or not found_bottom):
        top_center = (center+j) * center_spacing
        bottom_center = (center-j) * center_spacing
        if (found_top or top_center + begin > end or top_center - bin_width > time):
            found_top = True
        else:
            result.append(center+j)
        if (found_bottom or bottom_center < 0 or bottom_center + bin_width < time):
            found_bottom = True
        else:
            result.append(center-j)
        j += 1
    return result

### Old binning functionality
def binSeries(period, duration, series, time):
    # Set up global output
    global_series_bins = []
    global_delta = period/n_global_bins
    # Set up local output
    local_series_bins = []
    local_delta = local_bin_width * duration
    local_lambda = 2 * local_observation_width * duration / (n_local_bins - 1)
    begin_time = period/2 - local_observation_width * duration
    end_time = period/2 + local_observation_width * duration
    # Begin looping through all data
    min_index = 202
    max_index = -1
    min_center = 3000
    for i in range(0, len(time)):
        c_time = time[i]
        # Determine which global bin this entry falls into
        global_ind = math.floor(c_time/global_delta)
        global_series_bins = addToSeriesBins(global_series_bins, global_ind, series[i])
        # Determine all local bins the entry falls into
        # First check if the entry falls within the examined period 
        if (c_time < begin_time or c_time > end_time):
            continue
        # Shift the current time to 0 to account for different window
        c_time -= begin_time
        # Determine the nearest bin (Assumes it will be able to fall into the one below hence floor)
        nearest_center_ind = math.floor(c_time/local_lambda)
        min_center = min_center if min_center < nearest_center_ind else nearest_center_ind
        # Find the surrounding bins that it's also contained by
        bin_indices = determineContainingBins(nearest_center_ind, local_lambda, local_delta, begin_time, end_time, c_time)
        # Add value to all found bins
        for j in range(0, len(bin_indices)):
            min_index = bin_indices[j] if bin_indices[j] < min_index else min_index
            max_index = bin_indices[j] if bin_indices[j] > max_index else max_index
            local_series_bins = addToSeriesBins(local_series_bins, bin_indices[j], series[i])
    # Now take the median of all bins
    global_series_binned = []
    for i in range(0, len(global_series_bins)):
        global_series_binned.append(np.median(global_series_bins[i]))
    local_series_binned = []
    for i in range(0, len(local_series_bins)):
        local_series_binned.append(np.median(local_series_bins[i]))
    return {
        "global": global_series_binned,
        "local": local_series_binned
    }

def updateBins(series_bins, view, index, lc_det_val, lc_pdc_val, cent_val):
    while(len(series_bins[view]["lc_det"]) <= index):
        series_bins[view]["lc_det"].append([])
        series_bins[view]["lc_pdc"].append([])
        series_bins[view]["cent"].append([])
    series_bins[view]["lc_det"][index].append(lc_det_val)
    series_bins[view]["lc_pdc"][index].append(lc_pdc_val)
    series_bins[view]["cent"][index].append(cent_val)
    return series_bins

def padBinsToLength(series_bins):
    # Check local length
    if (len(series_bins["local"]["lc_det"]) < n_local_bins):
        series_bins = updateBins(series_bins, "local", n_local_bins-1, [], [], [])
    # Check global length
    if (len(series_bins["global"]["lc_det"]) < n_global_bins):
        series_bins = updateBins(series_bins, "global", n_global_bins-1, [], [], [])
    return series_bins

def binAllTimeSeries(period, duration, phase_folded_time, lc_detrend, lc_pdc, delta_r):
    # Begin to bin all the series
    # Initialise outputs
    series_bins = {
        "global": {
            "lc_det": [],
            "lc_pdc": [],
            "cent": []
        },
        "local": {
            "lc_det": [],
            "lc_pdc": [],
            "cent": []
        }
    }
    global_delta = period / n_global_bins
    local_delta = local_bin_width * duration
    local_lambda = 2 * local_observation_width * duration / (n_local_bins - 1)
    begin_time = period/2 - local_observation_width * duration
    end_time = period/2 + local_observation_width * duration
    # Begin looping through all data
    pdc_viable = len(lc_detrend) < len(lc_pdc)
    for i in range(0, len(phase_folded_time)):
        c_time = phase_folded_time[i]
        lc_det_val = lc_detrend[i]
        if (pdc_viable):
            lc_pdc_val = lc_pdc[i]
            cent_val = delta_r[i]
        else:
            lc_pdc_val = False
            cent_val = 0 
        # Determine which global bin this entry falls into
        global_ind = math.floor(c_time/global_delta)
        series_bins = updateBins(series_bins, "global", global_ind, lc_det_val, lc_pdc_val, cent_val)
        # Determine all the local bins that the entry falls into
        # First check if the entry falls within the examined period
        if (c_time < begin_time or c_time > end_time):
            continue
        # Shift the current times epoch to the beginning time to account for different window
        c_time -= begin_time
        # Determine the nearest bin (Assumes it will be able to fall into the one below, hence floor)
        nearest_center_ind = math.floor(c_time/local_lambda)
        # Find the surrounding bins that it's also contained by
        local_bin_indices = determineContainingBins(nearest_center_ind, local_lambda, local_delta, begin_time, end_time, c_time)
        # Add value to all found bins
        for j in range(0, len(local_bin_indices)):
            series_bins = updateBins(series_bins, "local", local_bin_indices[j], lc_det_val, lc_pdc_val, cent_val)

    # Check all bins are padded to length
    series_bins = padBinsToLength(series_bins)
    # Now begin to take the median of all bins
    series_output = {
        "global": {
            "lc_det": [],
            "lc_pdc": [],
            "cent": []
        },
        "local": {
            "lc_det": [],
            "lc_pdc": [],
            "cent": []
        }
    }
    global_nan_count = 0
    for i in range(0, len(series_bins["global"]["lc_det"])):
        lc_detrend_median = np.median(series_bins["global"]["lc_det"][i])
        if (np.isnan(lc_detrend_median)):
            global_nan_count += 1
        series_output["global"]["lc_det"].append(lc_detrend_median)
        if (pdc_viable):
            series_output["global"]["lc_pdc"].append(np.median(series_bins["global"]["lc_pdc"][i]))
        else:
            series_output["global"]["lc_pdc"].append(0)
        series_output["global"]["cent"].append(np.median(series_bins["global"]["cent"][i]))
    local_nan_count = 0
    for i in range(0, len(series_bins["local"]["lc_det"])):
        lc_detrend_median = np.median(series_bins["local"]["lc_det"][i])
        if (np.isnan(lc_detrend_median)):
            local_nan_count += 1
        series_output["local"]["lc_det"].append(np.median(series_bins["local"]["lc_det"][i]))
        if (pdc_viable):
            series_output["local"]["lc_pdc"].append(np.median(series_bins["local"]["lc_pdc"][i]))
        else:
            series_output["local"]["lc_pdc"].append(0)
        series_output["local"]["cent"].append(np.median(series_bins["local"]["cent"][i]))
    result = {
        "cent": {
            "global": series_output["global"]["cent"],
            "local": series_output["local"]["cent"]
        },
        "lc": {}
    }

    if ((local_nan_count > n_local_bins/2 or global_nan_count > n_global_bins/2) and pdc_viable):
        result["lc"]["global"] = series_output["global"]["lc_pdc"]
        result["lc"]["local"] = series_output["local"]["lc_pdc"]
    else:
        result["lc"]["global"] = series_output["global"]["lc_det"]
        result["lc"]["local"] = series_output["local"]["lc_det"]

    return result

def prepareBinnedTimeSeries(period, duration, epoch, lc_data, dv_data):
    # First extract series from DV file
    # Time is identical for each file
    time = dv_data["TIME"]
    if ("LC_DETREND" in dv_data.columns.names):
        lc_detrend = dv_data["LC_DETREND"]
    elif ("LC_INIT" in dv_data.columns.names):
        lc_detrend = dv_data["LC_INIT"]
    else:
        # At least one DV file seems to have the structure of an original fits file?
        lc_detrend = dv_data["PDCSAP_FLUX"]
    dv_nans = np.isnan(lc_detrend)
    lc_detrend = lc_detrend[np.logical_not(dv_nans)]
    time = time[np.logical_not(dv_nans)]
    # Then extract series from LC file
    lc_pdc = lc_data["PDCSAP_FLUX"]
    cent_x = lc_data["MOM_CENTR1"]
    cent_y = lc_data["MOM_CENTR2"]
    lc_nans = np.isnan(lc_pdc)
    lc_pdc = lc_pdc[np.logical_not(lc_nans)]
    cent_x = cent_x[np.logical_not(lc_nans)]
    cent_y = cent_y[np.logical_not(lc_nans)]
    x_shift = cent_x - np.median(cent_x)
    y_shift = cent_y - np.median(cent_y)
    delta_r = []
    for i in range(0, len(x_shift)):
        delta_r.append(math.sqrt(x_shift[i]**2 + y_shift[i]**2))
    # Shift epochs of data
    time -= epoch
    # Phase fold time
    phase_folded_time = (time + period/2) % period
    ################################################################################
    # Remove anomalies of > 3.5sigma difference from surrounding points
    # What does surrounding mean??
    ################################################################################
    # Median bin to local and global views
    return [binAllTimeSeries(period, duration, phase_folded_time, lc_detrend, lc_pdc, delta_r), phase_folded_time]

def determineTransitIndices(duration, period):
    middle_global_index = n_global_bins/2
    global_proportion = duration / period
    min_global_index = math.ceil(middle_global_index - (global_proportion/2 * n_global_bins))
    max_global_index = math.floor(middle_global_index + (global_proportion/2 * n_global_bins))
    middle_local_index = n_local_bins/2
    local_proportion = 1/8
    min_local_index = math.ceil(middle_local_index - (local_proportion/2 * n_local_bins))
    max_local_index = math.floor(middle_local_index + (local_proportion/2 * n_local_bins))
    return [min_global_index, max_global_index, min_local_index, max_local_index]

def normaliseBinnedTimeSeries(duration, period, detected_depth, binned_series):
    [min_global_index, max_global_index, min_local_index, max_local_index] = determineTransitIndices(duration, period)
    out_of_transit_lc = {
        "local": np.concatenate((binned_series["lc"]["local"][:min_local_index], binned_series["lc"]["local"][max_local_index:])),
        "global": np.concatenate((binned_series["lc"]["global"][:min_global_index], binned_series["lc"]["global"][max_global_index:]))
    }
    out_of_transit_cent = {
        "local": np.concatenate((binned_series["cent"]["local"][:min_local_index], binned_series["cent"]["local"][max_local_index:])),
        "global": np.concatenate((binned_series["cent"]["global"][:min_global_index], binned_series["cent"]["global"][max_global_index:]))
    }
    out_of_transit_lc_nans_removed = {
        "local": out_of_transit_lc["local"][np.logical_not(np.isnan(out_of_transit_lc["local"]))],
        "global": out_of_transit_lc["global"][np.logical_not(np.isnan(out_of_transit_lc["global"]))]
    }
    out_of_transit_cent_nans_removed = {
        "local": out_of_transit_cent["local"][np.logical_not(np.isnan(out_of_transit_cent["local"]))],
        "global": out_of_transit_cent["global"][np.logical_not(np.isnan(out_of_transit_cent["global"]))]
    }
    binned_series["lc"]["local"] -= np.median(out_of_transit_lc_nans_removed["local"])
    binned_series["lc"]["global"] -= np.median(out_of_transit_lc_nans_removed["global"])
    binned_series["cent"]["local"] -= np.median(out_of_transit_cent_nans_removed["local"])
    binned_series["cent"]["global"] -= np.median(out_of_transit_cent_nans_removed["global"])
    binned_series["lc"]["local"] /= detected_depth
    binned_series["lc"]["global"] /= detected_depth
    return binned_series

def correctBinnedNans(duration, period, binned_series):
    [min_global_index, max_global_index, min_local_index, max_local_index] = determineTransitIndices(duration, period)
    for i in range(0, len(binned_series["lc"]["local"])):
        if (i <= min_local_index or i >= max_local_index):
            binned_series["lc"]["local"][i] = 0 if np.isnan(binned_series["lc"]["local"][i]) else binned_series["lc"]["local"][i]
            binned_series["cent"]["local"][i] = 0 if np.isnan(binned_series["cent"]["local"][i]) else binned_series["cent"]["local"][i]
    for i in range(0, len(binned_series["lc"]["global"])):
        if (i <= min_global_index or i >= max_global_index):
            binned_series["lc"]["global"][i] = 0 if np.isnan(binned_series["lc"]["global"][i]) else binned_series["lc"]["global"][i]
            binned_series["cent"]["global"][i] = 0 if np.isnan(binned_series["cent"]["global"][i]) else binned_series["cent"]["global"][i]
    return linearlyInterpolateBinNans(binned_series)

def interpolateSeries(series):
    # Correct for first value incase that's Nan
    if (np.isnan(series[0])):
        i = 1
        while (np.isnan(series[i])):
            i += 1
            ### This would have gone wrong before hand if this occurs
            if (i == len(series)):
                print("Something's gone seriously wrong for this TCE!")
                return series
        val = series[i]
        for j in range(0, i):
            series[j] = val

    for i in range(1, len(series)):
        if (np.isnan(series[i])):
            before_val = series[i-1]
            j = i+1
            reach_end = False
            while (np.isnan(series[j])):
                j += 1
                if (j == len(series)):
                    reach_end = True
                    break
            if reach_end:
                for j in range(i, len(series)):
                    series[j] = before_val
                return series
            after_val = series[j]
            delta = (after_val - before_val) / (j - i + 1)
            for k in range(i, j):
                pos =  k - i + 1
                series[k] = before_val + delta * pos
    return series

def linearlyInterpolateBinNans(binned_series):
    binned_series["lc"]["local"] = interpolateSeries(binned_series["lc"]["local"])
    binned_series["lc"]["global"] = interpolateSeries(binned_series["lc"]["global"])
    binned_series["cent"]["local"] = interpolateSeries(binned_series["cent"]["local"])
    binned_series["cent"]["global"] = interpolateSeries(binned_series["cent"]["global"])
    return binned_series

def normaliseSeriesForML(series):
    series_usable = True
    max_val = max(series)
    min_val = min(series)
    delta = max_val - min_val
    series -= min_val
    series /= delta
    fin_max_val = max(series)
    fin_min_val = min(series)
    if (fin_max_val != 1 or fin_min_val != 0):
        series_usable = False
    return [series, series_usable]

def finalNormalisation(binned_series):
    [binned_series["lc"]["local"], lc_local_valid] = normaliseSeriesForML(binned_series["lc"]["local"])
    [binned_series["lc"]["global"], lc_global_valid] = normaliseSeriesForML(binned_series["lc"]["global"])
    tce_usable = lc_local_valid and lc_global_valid
    return [binned_series, tce_usable]


def calculateExpectedDuration(planet_to_star_radius_ratio, orbital_period, stellar_density):
    scaled_stellar_density = stellar_density * 1408
    scaled_orbital_period = orbital_period * 24 * 60 * 60
    first_half = 2 * ( 1 + planet_to_star_radius_ratio )
    numerator = 3 * scaled_orbital_period
    denominator = 8 * math.pow(math.pi, 2) * scientific_constants.G * scaled_stellar_density
    second_half = (numerator / denominator) ** (1.0/3)
    expected_dur_seconds = first_half * second_half
    expected_dur_days = expected_dur_seconds / 24 / 60 / 60
    return expected_dur_days

def collateParameters(tce_id, tce_data, headers, period, duration, pc):
    # Parameters still required
    # - max MES to exp MES (from SES and #transits)
    # Parameters to check calculation with Oscar for
    # - SNR - Don't think this is right still?
    # - Log duration over the expected duration from stellar density and orbital period - check calculation's correct
    # - Band magnitude - check this is correct header value
    event_parameters = {
        "tce_id": tce_id,
        "tic_id": int(tce_data["ticid"]),
        "pc": (1 if pc else 0)
    }
    # Orbit fit parameters
    event_parameters["semi_major_scaled_to_stellar_radius"] = float(tce_data["ratioSemiMajorAxisToStarRadius"])
    transit_count = int(tce_data["expectedtransitcount"])
    event_parameters["number_of_transits"] = transit_count
    event_parameters["transit_depth"] = float(tce_data["transitDepthPpm"])
    event_parameters["ingress_duration"] = float(tce_data["transitIngressTimeHours"])
    event_parameters["impact_parameter"] = float(tce_data["minImpactParameter"])
    event_parameters["signal_to_noise_ratio"] = float(tce_data["maxses"])
    event_parameters["ratio_of_mes_to_expected_mes"] = float(tce_data["mes"]) / (float(tce_data["maxses"]) * math.sqrt(transit_count)) if transit_count > 0 else 0
    # "Planetary" parameters
    event_parameters["ratio_of_planet_to_star_radius"] = float(tce_data["ratioPlanetRadiusToStarRadius"])
    event_parameters["log_ratio_of_planet_to_earth_radius"] = math.log(float(tce_data["planetRadiusEarthRadii"])) # Worth noting they do over 13Re perhaps due to a lot of exoplanets being hot jupiters?
    # Stellar parameters
    event_parameters["band_magnitude"] = headers["TESSMAG"]
    event_parameters["stellar_radius"] = headers["RADIUS"]
    event_parameters["total_proper_motion"] = headers["PMTOTAL"]
    event_parameters["stellar_log_g"] = headers["LOGG"]
    event_parameters["stellar_melaticity"] = headers["MH"]
    event_parameters["effective_temperature"] = headers["TEFF"]
    if (False and event_parameters["stellar_density"]):
        expected_duration = calculateExpectedDuration(event_parameters["ratio_of_planet_to_star_radius"], period, event_parameters["stellar_density"])
        event_parameters["log_duration_over_expected_duration"] = math.log(duration / expected_duration)
    else:
        event_parameters["log_duration_over_expected_duration"] = 0

    return event_parameters

def normalRound(val):
    return math.floor(val) if ((val - math.floor(val)) < 0.5) else math.ceil(val)

# More precise values are always found in the TOI file hence the comparison function structure
def compareValues(tce_val, toi_val, tce_id):
    if (np.isnan(tce_val) or np.isnan(toi_val)):
        return False
    dp = 0
    while (tce_val % 1 != 0):
        dp += 1
        tce_val = round(tce_val, 8)
        tce_val *= 10
    toi_val *= pow(10, dp)
    return tce_val == normalRound(toi_val) or tce_val == round(toi_val)

# Define TCE comparison function
# This list is definitely not an exhaustive list of properties but I think it's sufficient
key_tuplets = [["transitEpochBtjd", "Epoch Value", "e"], ["transitDepthPpm", "Transit Depth Value", "de"], ["transitDurationHours", "Transit Duration Value", "du"], ["orbitalPeriodDays", "Orbital Period Value", "o"]]
def matchTCE(tce_data, toi_data):
    tce_id = tce_data["tceid"]
    factor_count = 0

    for [tceKey, toiKey, mapKey] in key_tuplets:
        tce_val = tce_data[tceKey]
        toi_val = toi_data[toiKey]
        similar = compareValues(tce_val, toi_val, tce_id)
        if similar:
            factor_count += 1

    # From testing it was determined if 3 or more factors were matched it was going to be the same event
    return factor_count >= 2

# Need to check all possible TOIs and there may be multiple per TIC
def determineCandidateStatus(tce_data, toi_data):
    tic_id = tce_data["ticid"]
    toi_data = toi_data[toi_data["TIC"] == tic_id]
    for index, toi in toi_data.iterrows():
        if matchTCE(tce_data, toi):
            return True

    return False

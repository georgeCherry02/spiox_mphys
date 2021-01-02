import math
import numpy as np
from scipy import constants as scientific_constants

def normaliseSeries(series):
    med = np.median(series)
    series = series / med - 1
    sd = np.std(series)
    series /= sd
    return series

def normaliseAndQualityCorrectData(dv_data, centroid_data, epoch):
    time = dv_data["TIME"]
    lc_detrend = dv_data["LC_DETREND"]
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

def addToSeriesBins(bins, index, value):
    while (len(bins) <= index):
        bins.append([])
    bins[index].append(value)
    return bins

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

def collateParameters(tce_id, tce_data, tic_data, headers, period, duration, pc):
    # Parameters still required
    # - max MES to exp MES (from SES and #transits)
    # Parameters to check calculation with Oscar for
    # - SNR - Don't think this is right still?
    # - Log duration over the expected duration from stellar density and orbital period - check calculation's correct
    # - Band magnitude - check this is correct header value
    event_parameters = {
        "tce_id": tce_id,
        "tic_id": int(tce_data["ticid"]),
        "pc": pc
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
    event_parameters["stellar_density"] = tic_data["rho"]
    if (event_parameters["stellar_density"]):
        expected_duration = calculateExpectedDuration(event_parameters["ratio_of_planet_to_star_radius"], period, event_parameters["stellar_density"])
        event_parameters["log_duration_over_expected_duration"] = math.log(duration / expected_duration)
    else:
        event_parameters["log_duration_over_expected_duration"] = 0

    return event_parameters

def determineCandidateStatus(tic_id, toi_data):
    return tic_id in toi_data["TIC"]

from astropy.io import fits
import glob
import numpy as np
import os
import pandas as pd

class FileHandler:

    expected_keys = ["tce_id", "tic_id", "band_magnitude", "effective_temperature", "impact_parameter", "ingress_duration", "log_ratio_of_planet_to_earth_radius", "log_duration_over_expected_duration", "number_of_transits", "ratio_of_mes_to_expected_mes", "ratio_of_planet_to_star_radius", "semi_major_scaled_to_stellar_radius", "signal_to_noise_ratio", "stellar_density", "stellar_log_g", "stellar_melaticity", "stellar_radius", "total_proper_motion", "transit_depth"]

    def __init__(self, output_dir, input_dir, tce_data_dir):
        self.input_dir = input_dir
        self.tce_data_dir = tce_data_dir
        # Make directory for light curves to be written into
        self.output_dir = output_dir + "processed_lcs_and_centroids/"
        if (not os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)
        # Initialise csv for collated scientific domain information
        self.collated_parameter_file = output_dir + "collated_scientific_domain_parameters.csv"
        output = self.collateKeysToCSVHeader()
        with open(self.collated_parameter_file, "w") as f:
            f.write(output)

    def collateKeysToCSVHeader(self):
        output = ""
        for i in range(0, len(self.expected_keys)):
            output += self.expected_keys[i]+", "
        output = output[0:len(output)-2]+"\n"
        return output

    def appendParameters(self, event_parameters):
        output = f''
        for i in range(0, len(self.expected_keys)):
            output += f'{event_parameters[self.expected_keys[i]]}, '
        output = output[0:len(output)-2]+"\n"
        with open(self.collated_parameter_file, "a") as f:
            f.write(output)

    def writeProcessedCurves(self, tce_id, binned_lc, binned_centroid):
        local_output_path = self.output_dir + tce_id + "-local"
        local_output_data = np.array([binned_lc["local"], binned_centroid["local"]])
        global_output_path = self.output_dir + tce_id + "-global"
        global_output_data = np.array([binned_lc["global"], binned_centroid["global"]])
        np.save(local_output_path, local_output_data)
        np.save(global_output_path, global_output_data)

    def parseDVFilePath(self, fp):
        sector_begin = fp.index("-")
        sector_end = fp.index("-", sector_begin+1)
        sector_begin += 1
        sector = fp[sector_begin:sector_end]
        tic_begin = sector_end
        tic_end = fp.index("-", tic_begin+1)
        tic_begin += 1
        tic_id = fp[tic_begin:tic_end]
        tce_id = fp[tic_begin:tic_end+3]
        while (tce_id[0] == '0'):
            tce_id = tce_id[1:len(tce_id)]
        return [tic_id, tce_id, sector]       

    def readInTCEData(self, tce_id, sector):
        # Look for sector information
        tce_data_path = glob.glob(self.tce_data_dir+"*"+sector+"*"+sector+"*.csv")
        tce_data = pd.read_csv(tce_data_path[0], skiprows=6)
        while (len(tce_id) < 14):
            tce_id = "0"+tce_id
        specific_tce_data = tce_data.loc[tce_data["tceid"] == tce_id]
        return specific_tce_data

    def readInDVFile(self, file_path):
        return fits.open(file_path)

    def readInCentroidData(self, tic_id, sector):
        centroid_path = glob.glob(self.input_dir+"*/*-"+sector+"-"+tic_id+"-0120-s_lc.fits")[0]
        return fits.open(centroid_path)[1].data

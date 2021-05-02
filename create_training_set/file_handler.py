from astropy.io import fits
import glob
import numpy as np
import os
import pandas as pd

class FileHandler:

    expected_keys = ["tce_id", "tic_id", "band_magnitude", "effective_temperature", "impact_parameter", "ingress_duration", "log_ratio_of_planet_to_earth_radius", "log_duration_over_expected_duration", "number_of_transits", "ratio_of_mes_to_expected_mes", "ratio_of_planet_to_star_radius", "semi_major_scaled_to_stellar_radius", "signal_to_noise_ratio", "stellar_log_g", "stellar_melaticity", "stellar_radius", "total_proper_motion", "transit_depth", "pc"]

    def __init__(self, dv_input_dir, lc_input_dir, output_dir, tce_data, toi_data, skipping):
        # Store directories
        self.dv_input_dir = dv_input_dir
        self.lc_input_dir = lc_input_dir
        self.tce_data = tce_data
        self.toi_data = toi_data
        # Make directory for light curves to be written into
        self.output_dir = output_dir + "processed_lcs_and_centroids/"
        if (not os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)
        # Initialise csv for collated scientific domain information
        self.collated_parameter_file = output_dir + "collated_scientific_domain_parameters.csv"
        if (not skipping):
            output = self.collateKeysToCSVHeader()
            with open(self.collated_parameter_file, "w") as f:
                f.write(output)
        # Keep track of skipped tces 
        self.skipped_tce_csv = output_dir + "skipped_tces.csv"
        if (not skipping):
            with open(self.skipped_tce_csv, "w") as f:
                f.write("skipped_tces\n")

    def collateKeysToCSVHeader(self):
        output = ""
        for i in range(0, len(self.expected_keys)):
            output += self.expected_keys[i]+","
        output = output[0:len(output)-1]+"\n"
        return output

    def appendParameters(self, event_parameters):
        output = f''
        for i in range(0, len(self.expected_keys)):
            output += f'{event_parameters[self.expected_keys[i]]},'
        output = output[0:len(output)-1]+"\n"
        with open(self.collated_parameter_file, "a") as f:
            f.write(output)

    def readInTCEData(self):
        return self.readInDataFile(True, 6)
    
    def readInTOIData(self):
        return self.readInDataFile(False, 4)

    def readInDataFile(self, use_tce, offset):
        return pd.read_csv((self.tce_data if use_tce else self.toi_data), skiprows=offset)

    def loadRawData(self, tic_id, sector):
        # Define file patterns
        if (False):
            ### File patterns for laptop
            dv_file_pattern = self.dv_input_dir+"*/*"+str(tic_id)+"*_dvt.fits"
            lc_file_pattern = self.lc_input_dir+"*/*"+str(tic_id)+"*_lc.fits"
        ### File patterns for glamdring
        dv_dir_pattern = "*-s*"+sector+"-*"+str(tic_id)+"-*-s/"
        main_file_pattern = "*"+str(tic_id)+"*"
        dv_file_pattern = self.dv_input_dir+dv_dir_pattern+main_file_pattern+"_dvt.fits"
        lc_file_pattern = self.lc_input_dir+main_file_pattern+"_lc.fit"
        if (int(sector) == 15):
            # Why is this the case??
            lc_file_pattern += "s"

        dv_file_paths = glob.glob(dv_file_pattern)
        lc_file_paths = glob.glob(lc_file_pattern)
        if ((len(dv_file_paths) != 1) or (len(lc_file_paths) != 1)):
            if (len(dv_file_paths) < 1):
                print("DV < 1")
                return [False, False]
            if (len(dv_file_paths) > 1):
                print("DV > 1")
            if (len(lc_file_paths) < 1):
                print("LC < 1")
                return [False, False]
            if (len(lc_file_paths) > 1):
                print("LC > 1")
        dv_hdul = fits.open(dv_file_paths[0])
        lc_hdul = fits.open(lc_file_paths[0])
        return [dv_hdul, lc_hdul]

    def writeProcessedCurves(self, tce_id, binned_lc, binned_centroid):
        local_output_path = self.output_dir + tce_id + "-local"
        local_output_data = np.array([binned_lc["local"], binned_centroid["local"]])
        global_output_path = self.output_dir + tce_id + "-global"
        global_output_data = np.array([binned_lc["global"], binned_centroid["global"]])
        np.save(local_output_path, local_output_data)
        np.save(global_output_path, global_output_data)

    def decideSkip(self, tce_id):
        # Output file path
        expected_path = self.output_dir + tce_id
        existing_paths = glob.glob(expected_path+"*.npy")
        return len(existing_paths) >= 2
    
    def appendSkippedTCE(self, tce_id):
        with open(self.skipped_tce_csv, "a") as f:
            f.write(tce_id+"\n")

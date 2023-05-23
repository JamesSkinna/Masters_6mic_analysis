"""Specify a directory -> Iterate through all flac files (in any subdirectories) and:
    1. Pre-process
    2. Run through ODAS
    3. Run through BirdNET
    4. Results files stored in 'results' directory (folder structure maintained)
    
***SEE: early_analysis.ipynb for further info"""

import os
import subprocess
import json
from datetime import datetime

import sox
from jinja2 import Environment, FileSystemLoader
from birdnetlib.batch import DirectoryMultiProcessingAnalyzer
from birdnetlib.analyzer import Analyzer


# Define constants
DIR_PATH = "data/raw/manicore/IC1/13-05-23/RPiID-000000004e8ff25b/2023-04-17"    # Directory to analyse
CONFIG_PATH = "src/configs"     # Where our ODAS config templates are located
LOCATION_DICT = {"manicore": {"lat": -5.750849, "long":-61.421894},
                 "silwood": {"lat": 51.409111, "long":-0.637820}}
# Set the location of data...
if "manicore" in DIR_PATH:
    LOCATION = "manicore"
elif "silwood" in DIR_PATH:
    LOCATION = "silwood"
else:
    LOCATION = "lab"


def convert_flac_to_wav(input_path):
    """Take flac file from data/raw directory
    --> Convert to equivalent .RAW file
    --> Store in new directory, inside data/processed
    Returns: File path of processed RAW file"""

    folder_path = input_path.split(".")[0]       # New folder path
    folder_path = folder_path.replace("raw", "processed")     # Move from raw folder to processed

    # Setup a new folder for the processed data
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # CONVERT TO RAW------------------------------------------------------------------------
    # Input FLAC file
    input_file = input_path

    # Output RAW audio file
    output_path = folder_path + "/raw_6_chan.raw"

    # Create a Sox object
    tfm = sox.Transformer()

    tfm.set_input_format(rate=16000, channels=6, bits=32, encoding="signed-integer")

    tfm.set_output_format(rate=16000, channels=6, bits=32, encoding="signed-integer")

    # Convert the FLAC file to RAW audio format
    tfm.build(input_file, output_path)

    return output_path, folder_path


def create_cfg_file(input_path, folder_path):
    """Sets up an ODAS config file to process RAW data file
    Returns: Path to config file"""
    # Create a Jinja2 environment with the file system loader
    env = Environment(loader=FileSystemLoader(CONFIG_PATH))

    # Load the template from disk
    template = env.get_template("6mic_post_analysis_template.cfg")

    # Setup the paths
    raw_input_path = input_path

    # Render the template with some data
    result = template.render(input_path=raw_input_path, folder_path=folder_path)

    # # Print the result
    # print(result)

    # Save the rendered config to file
    config_path = folder_path + "/config.cfg"
    with open(config_path, "w") as file:
        file.write(result)

    return config_path


def odas_process(config_path):
    """Runs ODAS (as a subprocess, e.g., on Linux terminal) 
    based on the given config file"""

    run_odas_cmd = "../../odas/build/bin/odaslive -c " + config_path

    subprocess.call(run_odas_cmd, shell=True)       # Python will wait for execution to complete before continuing
    # Note: shell=True is important for searching relative file paths


def extract_separate_channels(folder_path):
    """Iterates through separated and postfiltered files (.RAW) of a directory
    --> These are the results from running ODAS
    Creates a new MP3 file for each channel
    --> 4 beamforming directions, thus, 8 new MP3 files (4 channels each for sep and pf)"""

    # RAW input files...
    sep_input_file = folder_path + "/sep.raw"
    pf_input_file = folder_path + "/pf.raw"

    for chan_num in range(1, 5):

        # Extract invidiual channels...
        tfm = sox.Transformer()
        remix_dict = {1: [chan_num]}    # Chan 1 of new file (output) will have channel chan_num (from input) - other channels are empty
        tfm.remix(remix_dict)           # Extract the correct channel

        # Output MP3 audio files...
        sep_output_file = folder_path + f"/sep_chan{str(chan_num)}.mp3"
        pf_output_file = folder_path + f"/pf_chan{str(chan_num)}.mp3"

        # 4 channels - 1 for each beamformed direction
        tfm.set_input_format(rate=16000, channels=4, bits=32, encoding="signed-integer")

        tfm.set_output_format(rate=16000, channels=1, bits=32, encoding="signed-integer")

        # Convert the RAW audio files
        tfm.build(sep_input_file, sep_output_file)
        tfm.build(pf_input_file, pf_output_file)


def extract_1_chan_from_flac(flac_path, folder_path):
    """Extracts the first channel from the original recording
    --> Converts and saves as MP3 file"""

    #Input FLAC file
    input_file = flac_path

    # Output RAW audio file
    output_file = folder_path + "/original_chan1.mp3"

    # Create a Sox object
    tfm = sox.Transformer()

    remix_dict = {1: [1]}    # Chan 1 of new file (output) will have channel 1 from input - other channels are empty
    tfm.remix(remix_dict)    # Extract the 1st channel

    tfm.set_input_format(rate=16000, channels=6, bits=32, encoding="signed-integer")

    tfm.set_output_format(rate=16000, channels=1, bits=32, encoding="signed-integer")

    # Convert the RAW file to MP3 audio format
    tfm.build(input_file, output_file)


def clean_up_processed_dir(folder_path):
    """Deletes all .raw files from a folder
    --> This frees up space, as .raw files are large, but no longer required"""

    for del_root, del_dirs, del_files in os.walk(folder_path):
        for del_name in del_files:
            if del_name.endswith(".raw"):
                del_file_path = os.path.join(del_root, del_name)
                os.remove(del_file_path)


def birdnet_process_dir(folder_path):
    """Runs BirdNET analysis for all mp3 files in a given directory
    --> Processes several files simultaneously (multiprocessing)"""

    results_dict = {}   # Dictionary to save all recording data


    def on_analyze_directory_complete(recordings):
        """This function runs once analysis is complete
        --> Simply collates & exports the results"""

        # Iterate through all detections - collate into 1 dictionary
        for recording in recordings:
            if recording.error:
                print("Error: ", recording.error_message)
            else:
                chan_name = recording.path.split("/")[-1]   # Extract just the file name (i.e., remove the path)
                results_dict[chan_name] = recording.detections  # Add results to entire dictionary

        # Write the detection results to a JSON file--------------------------------
        # Specify the file path to write the detections
        results_path = folder_path + '/results.json'

        # Write the BirdNET detections to the file
        with open(results_path, 'w') as file:
            json.dump(results_dict, file, indent=4)


    analyzer = Analyzer()

    directory = folder_path
    # Setup analyser parameters...
    if LOCATION == "lab":       # Don't specify lat and long for lab tests
        batch = DirectoryMultiProcessingAnalyzer(
            directory,
            analyzers=[analyzer],
            date=datetime(year=2023, month=5, day=10), # use date or week_48
            min_conf=0.4,       # Set a minimum confidence of 0.4 (for probability)
        )
    else:                       # Specify lat and long
        batch = DirectoryMultiProcessingAnalyzer(
            directory,
            analyzers=[analyzer],
            lat=LOCATION_DICT[LOCATION]["lat"],            # Manicore Lat & Long
            lon=LOCATION_DICT[LOCATION]["long"],
            date=datetime(year=2023, month=5, day=10), # use date or week_48
            min_conf=0.4,       # Set a minimum confidence of 0.4 (for probability)
        )

    # Specify function to run once analysis is complete
    batch.on_analyze_directory_complete = on_analyze_directory_complete
    # Process our analysis (for the entire directory)
    batch.process()




# Run our processing for all flac files in a given directory-------------------------------
for root, dirs, files in os.walk(DIR_PATH, topdown=False):
    for name in files:
        if name.endswith(".flac"):

            original_file_path = os.path.join(root, name)
            print(f"processing: {original_file_path}")

            print("Converting to RAW...")
            RAW_file_path, processed_folder_path = convert_flac_to_wav(original_file_path)

            print("Processing through ODAS...")
            current_config_path = create_cfg_file(RAW_file_path, processed_folder_path)
            odas_process(current_config_path)

            print("ODAS complete. Extracting channels to MP3...")
            extract_separate_channels(processed_folder_path)
            extract_1_chan_from_flac(original_file_path, processed_folder_path)

            print("MP3 extraction complete. Deleting '.raw' files.")
            clean_up_processed_dir(processed_folder_path)

            print("Processing through BirdNET...")
            birdnet_process_dir(processed_folder_path)
            print("BirdNET processing complete!")

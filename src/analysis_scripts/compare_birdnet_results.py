"""Specify a results directory --> Iterate through all results files. For each:

1. Compare confidence levels - boxplots for different species
    --> Original, single channel VS most confident beamformed detection
    --> Here, we only look for detections present in both original & beamformed channelss
        --> i.e., same species, same time interval
        
2. Compare counts - bar charts for each species
    --> To be safe, set a very high minium confidence level of 0.7"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define constants
DIR_PATH = "data/processed/speaker_sphere_lab_tests"

# FUNCTIONS FOR PROCESSING A DIRECTORY OF RESULTS FILES-----------------------------------------------------------------------------------
def read_results_from_file(file_path):
    """Read the BirdNET detections from a file"""

    with open(file_path, 'r') as file:
        results_dict = json.load(file)

    return results_dict


def extract_max_bf_conf(mono_detection, results_dict):
    """Checks a single detection from a mono recording 
    --> Identifies whether same detection is present in one or more of beamformed recordings
    --> Returns beamformed channel with highest confidence"""

    species_name = mono_detection["common_name"]         # Extract the bird name from the detection entry
    start_time = mono_detection["start_time"]            # Extract the start time from detection entry

    max_bf_conf = 0
    max_conf_chan = ""

    for channel in results_dict.keys():
        if "sep" in channel:                        # Iterate through all beamformed (bf) channels
            bf_chan_data = results_dict[channel]
            for bf_detection in bf_chan_data:       # Iterate through all detections in bf channel
                if (bf_detection["common_name"] == species_name) and (bf_detection["start_time"] == start_time):        # i.e., if looking at the same detection instance
                    if bf_detection["confidence"] > max_bf_conf:
                        max_bf_conf = bf_detection["confidence"]    # Update highest confidence
                        max_conf_chan = channel

    return max_bf_conf, max_conf_chan


def update_species(species_dict, species_name, conf, start_time, max_conf_chan=""):
    """Updates the data of an existing species, in a dictionary"""

    species_dict[species_name]["count"] += 1
    species_dict[species_name]["conf_sum"] += conf
    species_dict[species_name]["max_conf"] = max(conf, species_dict[species_name]["max_conf"])
    species_dict[species_name]["conf_avg"] = round(species_dict[species_name]["conf_sum"] / species_dict[species_name]["count"], 2)
    species_dict[species_name]["conf_list"].append(conf)
    species_dict[species_name]["start_time_list"].append(start_time)
    if max_conf_chan != "":
        species_dict[species_name]["max_conf_chan_list"].append(max_conf_chan)

    return species_dict


def add_new_species(species_dict, species_name, conf, start_time, max_conf_chan=""):
    """Adds data for a new species, to a dictionary"""

    species_dict[species_name] = {}       # Create new nested dictionary
    species_dict[species_name]["count"] = 1
    species_dict[species_name]["conf_sum"] = conf
    species_dict[species_name]["max_conf"] = conf
    species_dict[species_name]["conf_avg"] = round(species_dict[species_name]["conf_sum"] / species_dict[species_name]["count"], 2)
    species_dict[species_name]["conf_list"] = [conf]
    species_dict[species_name]["start_time_list"] = [start_time]
    if max_conf_chan != "":
        species_dict[species_name]["max_conf_chan_list"] = [max_conf_chan]

    return species_dict


def add_detection_to_results(mono_detection, max_bf_conf, max_conf_chan, overall_results):
    """Adds data from a detection (that occurred in both original & beamformed channels)
    ...to the overall processed results"""

    mono_results = overall_results["mono_channel"]
    bf_results = overall_results["beamformed"]

    species_name = mono_detection["common_name"]
    start_time = mono_detection["start_time"]
    mono_conf = mono_detection["confidence"]

    if species_name in mono_results:       # Check if species is present in our overall dictionary
        overall_results["mono_channel"] = update_species(mono_results, species_name, mono_conf, start_time)
        overall_results["beamformed"] = update_species(bf_results, species_name, max_bf_conf, start_time, max_conf_chan)
    else:
        overall_results["mono_channel"] = add_new_species(mono_results, species_name, mono_conf, start_time)
        overall_results["beamformed"] = add_new_species(bf_results, species_name, max_bf_conf, start_time, max_conf_chan)

    return overall_results


def process_results_dict(results_dict, overall_results):
    """Processes a single results dictionary (i.e., results from one file)
    --> Checks for detections present in both original & beamformed channels
    --> Identifies the beamformed channel with the highest conf level
    --> Adds these detections to overall results, storing:
        --> Species count
        --> Max conf level
        --> List of all conf levels
        --> List of occurrence start times
        --> Avg conf level per species
        --> Which beamformed channel had highest conf (indicating the likely direction)"""

    # Extract all detections from the mono channel (channel 1)
    mono_data = results_dict["original_chan1.mp3"]

    for mono_detection in mono_data:
        max_bf_conf, max_conf_chan = extract_max_bf_conf(mono_detection, results_dict)
        if max_bf_conf > 0:                 # If occurence in both original & beamformed channels
            overall_results = add_detection_to_results(mono_detection, max_bf_conf, max_conf_chan, overall_results)

    return overall_results


def write_processed_to_file(processed_dict):
    """After processing an entire directory of (odas/birdnet) results files
    --> Writes the final processed results to file"""

    # Specify the file path to write the detections
    processed_path = DIR_PATH + '/processed.json'

    # Write the BirdNET detections to the file
    with open(processed_path, 'w') as file:
        json.dump(processed_dict, file, indent=4)


# FUNCTIONS FOR PLOTTING PROCESSED RESULTS-----------------------------------------------------------------------------------
def set_box_color(bp, color):
    """Sets the colours of a single box, from a matplotlib boxplot"""
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def get_initials(species_name):
    """Extract the intials from a species' full name"""
    # Split the name into individual words
    words = species_name.split()

    # Extract the first character of each word
    initials = [word[0].upper() for word in words]

    # Return the initials as a string
    return ''.join(initials)


def setup_new_plot(xlabel, ylabel, title):
    """Initialises a new matplotlib plot, with desired parameters"""
    plt.figure(figsize=(18, 12))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)


def boxplot_conf_comparison(mono_data, bf_data, file_path):
    """Creates a boxplot to compare confidence levels of mono-channel vs beamformed
    --> Split into groups (1 group per species) - for side-by-side comparison"""

    title = "Confidence levels across species - Mono-channel vs Beamformed Recordings"
    setup_new_plot("Species Initials", "Confidence Levels", title)

    boxplot_conf_data_mono = []
    boxplot_conf_data_bf = []
    boxplot_labels = []

    # Extract the confidence levels list for each species - append to list of lists
    for species in mono_data.keys():
        if mono_data[species]["count"] >= 5:    # Only plot those with more than 5 detections (greater sample size)
            boxplot_conf_data_mono.append(mono_data[species]["conf_list"])
            boxplot_conf_data_bf.append(bf_data[species]["conf_list"])
            boxplot_labels.append(get_initials(species))

    # Plot the data - unfortunately, for side-by-side groups, we have to lay it out manually...
    bpl = plt.boxplot(boxplot_conf_data_mono, positions=np.array(range(len(boxplot_conf_data_mono)))*2.0-0.4, sym='', widths=0.6)
    bpr = plt.boxplot(boxplot_conf_data_bf, positions=np.array(range(len(boxplot_conf_data_bf)))*2.0+0.4, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Mono-channel')
    plt.plot([], c='#2C7BB6', label='Beamformed')
    plt.legend()

    plt.xticks(range(0, len(boxplot_labels) * 2, 2), boxplot_labels)
    plt.xlim(-2, len(boxplot_labels)*2)
    plt.ylim(0.4, 1)
    plt.tight_layout()

    plt.savefig(file_path)
    plt.show()




# RUN THE ANALYSIS----------------------------------------------------------------------------------------------------------
# New dict to compare single channel vs best of beamformed
processed_results = {"mono_channel": {},
                     "beamformed": {}}

# Analyse all results files in the directory
for root, dirs, files in os.walk(DIR_PATH, topdown=False):
    for name in files:
        if name == "results.json":
            results_path = os.path.join(root, name)
            current_results_dict = read_results_from_file(results_path)

            processed_results = process_results_dict(current_results_dict, processed_results)

write_processed_to_file(processed_results)

# Plot the data for visual inspection
mono_dict = processed_results["mono_channel"]
bf_dict = processed_results["beamformed"]
boxplot_file_path = DIR_PATH + "/boxcompare.png"

boxplot_conf_comparison(mono_dict, bf_dict, boxplot_file_path)

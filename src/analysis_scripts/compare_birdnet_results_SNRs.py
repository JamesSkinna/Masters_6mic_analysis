"""A slight modification to the compare_birdnet_results.py file - specifically, for the SNR lab tests

Specify a results directory --> Iterate through all results files. For each:
        
1. Compare counts - bar charts for each species
    --> Do this for all confidence thresholds, from 0.4 (min conf from birdnet) to 1, in steps of 0.05)
    --> Save bar_graphs of 0.4, 0.6 & 0.8 conf only"""
    
    

import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np

# Define constants
DIR_PATH = "data/processed/speaker_sphere_lab_tests/n15SNR"

# Set the location of data...
if "manicore" in DIR_PATH:
    LOCATION = "Manicore"
elif "silwood" in DIR_PATH:
    LOCATION = "Silwood"
else:
    LOCATION = "Lab: Speaker Sphere"

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


def update_species(species_dict, species_name, conf, start_time, file_path, max_conf_chan=""):
    """Updates the data of an existing species, in a dictionary"""

    species_dict[species_name]["conf_list"].append(conf)
    species_dict[species_name]["start_time_list"].append(start_time)
    if max_conf_chan != "":
        species_dict[species_name]["max_conf_chan_list"].append(max_conf_chan)
        species_dict[species_name]["mp3_identifier"].append(f"{file_path}, {max_conf_chan}, {str(start_time)}")
    else:
        species_dict[species_name]["mp3_identifier"].append(f"{file_path}, original_channel1.mp3, {str(start_time)}")

    return species_dict


def add_new_species(species_dict, species_name, conf, start_time, file_path, max_conf_chan=""):
    """Adds data for a new species, to a dictionary"""

    species_dict[species_name] = {}       # Create new nested dictionary
    species_dict[species_name]["conf_list"] = [conf]
    species_dict[species_name]["start_time_list"] = [start_time]
    if max_conf_chan != "":
        species_dict[species_name]["max_conf_chan_list"] = [max_conf_chan]
        species_dict[species_name]["mp3_identifier"] = [f"{file_path}, {max_conf_chan}, {str(start_time)}"]
    else:
        species_dict[species_name]["mp3_identifier"] = [f"{file_path}, original_channel1.mp3, {str(start_time)}"]

    return species_dict


def add_detection_to_results(mono_detection, max_bf_conf, max_conf_chan, overall_results, file_path):
    """Adds data from a detection (that occurred in both original & beamformed channels)
    ...to the overall processed results"""

    mono_results = overall_results["mono_channel"]
    bf_results = overall_results["beamformed"]

    species_name = mono_detection["common_name"]
    start_time = mono_detection["start_time"]
    mono_conf = mono_detection["confidence"]

    if species_name in mono_results:       # Check if species is present in our overall dictionary
        overall_results["mono_channel"] = update_species(mono_results, species_name, mono_conf, start_time, file_path)
        overall_results["beamformed"] = update_species(bf_results, species_name, max_bf_conf, start_time, file_path, max_conf_chan)
    else:
        overall_results["mono_channel"] = add_new_species(mono_results, species_name, mono_conf, start_time, file_path)
        overall_results["beamformed"] = add_new_species(bf_results, species_name, max_bf_conf, start_time, file_path, max_conf_chan)

    return overall_results


def process_results_dict(results_dict, overall_results, file_path):
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
            overall_results = add_detection_to_results(mono_detection, max_bf_conf, max_conf_chan, overall_results, file_path)

    return overall_results


def write_processed_to_file(processed_dict, file_name):
    """After processing an entire directory of (odas/birdnet) results files
    --> Writes the final processed results to file"""

    # Specify the file path to write the detections
    processed_path = DIR_PATH + "/" + file_name

    # Write the BirdNET detections to the file
    with open(processed_path, 'w') as file:
        json.dump(processed_dict, file, indent=4)


def add_detection_to_species_count(overall_results, detection, channel):
    """Updates the overall count, based on an individual detection"""
    species_name = detection["common_name"]

    if species_name in overall_results[channel]:         # Check if exists already in our count dictionary
        overall_results[channel][species_name] += 1
    else:
        # For new entries
        overall_results[channel][species_name] = 1       # Create new count

    return overall_results


def extract_unique_bf_detections(results_dict, conf_thresh):
    """Extracts all detections from the beamformed channels above a min conf threshold, 
    # ...and removes duplicate detections (e.g., same species & start_time on different channels)"""
    
    conf_detections = []    # New list, for all bf detections above threshold

    for channel in results_dict.keys():
        if "sep" in channel:                        # Iterate through all beamformed (bf) channels
            bf_chan_data = results_dict[channel]
            for bf_detection in bf_chan_data:       # Iterate through all detections in bf channel
                if bf_detection["confidence"] >= conf_thresh:
                    # Make a 'Primary Key' (to identify unique detections), from name & start time
                    prim_key = bf_detection["common_name"] + str(bf_detection["start_time"])
                    bf_detection["prim_key"] = prim_key
                    conf_detections.append(bf_detection)
    
    unique_detections = []              # New list, for all unique detections
    unique_detection_primary_keys = []  # List for storing unique keys

    for detection in conf_detections:
        if detection["prim_key"] not in unique_detection_primary_keys:      # i.e., if it's a new, unique detection
            unique_detections.append(detection)                             # Add unique detection info
            unique_detection_primary_keys.append(detection["prim_key"])     # Update primary key list

    return unique_detections


def count_detections_in_dict(results_dict, overall_results, conf_thresh):
    """Takes a dictionary (from a results.json file) and updates the overall count
    ...dictionary with species counts, above a certain conf threshold.
    --> Adds all detections from mono channel (above threshold)
    --> Adds unique detections from beamformed channels (checks to avoid duplicates)"""

    # Extract all detections from the mono channel (channel 1)
    mono_data = results_dict["original_chan1.mp3"]

    for mono_detection in mono_data:
        if mono_detection["confidence"] >= conf_thresh:
            overall_results = add_detection_to_species_count(overall_results, mono_detection, "mono_channel")

    # Extract detections from the beamformed channels (unique & above threshold)
    bf_data = extract_unique_bf_detections(results_dict, conf_thresh)

    for bf_detection in bf_data:
        overall_results = add_detection_to_species_count(overall_results, bf_detection, "beamformed")

    return overall_results


def add_conf_metrics(processed_dict):
    """Takes a dictionary of processed results
    --> Extracts each confidence list (each channel, each species)
    --> Caclulates and adds: mean, median, stdev, max & count"""

    for chan in processed_dict.keys():
        chan_data = processed_dict[chan]
        for species in chan_data.keys():
            species_data = chan_data[species]

            conf_list = species_data["conf_list"]
            mean = round(np.mean(conf_list), 3)
            median = round(np.median(conf_list), 3)
            stdev = round(np.std(conf_list), 3)
            max_conf = round(np.max(conf_list), 3)
            count = len(conf_list)

            processed_dict[chan][species]["conf_avg"] = mean
            processed_dict[chan][species]["conf_median"] = median
            processed_dict[chan][species]["conf_stdev"] = stdev
            processed_dict[chan][species]["conf_max"] = max_conf
            processed_dict[chan][species]["count"] = count

    return processed_dict


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


def draw_histograms(hist_data, species_names, n_rows, n_cols, title, file_path):
    """Draws several histograms in a single plot
    --> Organised in a grid of n_rows x n_cols"""

    fig=plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16)         # As we have subplots, we set an overall title with suptitle

    for i, name in enumerate(species_names):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        ax.hist(hist_data[i], bins=20)

        ax.set_ylabel("Frequency")
        ax.set_xlabel("Confidence")
        ax.grid(True)
        ax.set_title(name, fontsize=12)
    
    fig.tight_layout()  # Improves appearance a bit.
    plt.subplots_adjust(top=0.92, wspace=0.3, hspace=0.3)
    
    plt.savefig(file_path)
    # plt.show()


def find_best_grid_arrangement(num_graphs):
    """Calculate the number of rows and columns for the grid"""
    num_rows = math.ceil(math.sqrt(num_graphs))
    num_cols = math.ceil(num_graphs / num_rows)

    return num_rows, num_cols


def plot_confidence_histograms(mono_data, bf_data, mono_file_path, bf_file_path):
    """Creates two plots - each containing X histograms (one for each species present)
    --> Used to assess whether confidence data is normally distributed"""

    hist_conf_data_mono = []
    hist_conf_data_bf = []
    hist_labels = []

    # Extract the confidence levels list for each species - append to list of lists
    for species in mono_data.keys():
        if mono_data[species]["count"] >= 20:    # Only plot those with more than 20 detections (otherwise, histograms aren't useful)
            hist_conf_data_mono.append(mono_data[species]["conf_list"])
            hist_conf_data_bf.append(bf_data[species]["conf_list"])
            hist_labels.append(get_initials(species))

    if hist_labels:
        num_rows, num_cols = find_best_grid_arrangement(len(hist_labels))

        mono_title = f"Distributions of confidence levels, per species - Mono-channel - {LOCATION}"
        bf_title = f"Distributions of confidence levels, per species - Beamformed - {LOCATION}"

        draw_histograms(hist_conf_data_mono, hist_labels, num_rows, num_cols, mono_title, mono_file_path)
        draw_histograms(hist_conf_data_bf, hist_labels, num_rows, num_cols, bf_title, bf_file_path)
    else:
        print("No histogram data to plot!")


def boxplot_conf_comparison(mono_data, bf_data, file_path):
    """Creates a boxplot to compare confidence levels of mono-channel vs beamformed
    --> Split into groups (1 group per species) - for side-by-side comparison"""

    title = f"Confidence levels across species - Mono-channel vs Beamformed Recordings - {LOCATION}"
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

    if boxplot_labels:        # If we have some data to plot (otherwise, = [], which acts as False)
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
        # plt.show()
    else:
        print("No boxplot data to plot!")


def get_unique_species(arr):
    """Create a list of all unique species (from either channel)"""
    unique_species = []
    for dictionary in arr:            # Iterate through all dictionaries
        unique_species.extend(list(dictionary.keys()))        # Add all species to a new list
    unique_species = list(set(unique_species))          # Remove duplicates from new list

    return unique_species


def barchart_count_comparison(mono_data, bf_data, conf_thresh, file_path):
    """Creates a boxplot to compare species counts of mono-channel vs beamformed
    --> Split into groups (1 group per species) - for side-by-side comparison"""

    title = f"Species counts for detections above {str(conf_thresh)} confidence - Mono-channel vs Beamformed Recordings - {LOCATION}"
    setup_new_plot("Species Initials", "Species Count", title)

    count_data_list = [mono_data, bf_data]
    barchart_count_mono = []
    barchart_count_bf = []
    barchart_labels = []

    species_list = get_unique_species(count_data_list)

    # Extract count data to a list...
    for species in species_list:
        barchart_labels.append(get_initials(species))

        if species in mono_data:
            barchart_count_mono.append(mono_data[species])
        else:
            barchart_count_mono.append(0)

        if species in bf_data:
            barchart_count_bf.append(bf_data[species])
        else:
            barchart_count_bf.append(0)

    if barchart_labels:        # If we have some data to plot (otherwise, = [], which acts as False)
        # Plot the data - unfortunately, for side-by-side groups, we have to lay it out manually...
        plt.bar(np.array(range(len(barchart_count_mono)))*2.0-0.3, barchart_count_mono, width=0.6, color='#D7191C')
        plt.bar(np.array(range(len(barchart_count_bf)))*2.0+0.3, barchart_count_bf, width=0.6, color='#2C7BB6')

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='#D7191C', label='Mono-channel')
        plt.plot([], c='#2C7BB6', label='Beamformed')
        plt.legend()

        plt.xticks(range(0, len(barchart_labels) * 2, 2), barchart_labels)
        plt.xlim(-2, len(barchart_labels)*2)
        # plt.ylim(0.4, 1)
        plt.tight_layout()

        plt.savefig(file_path)
        # plt.show()
    else:
        print("No barchart data to plot!")


# RUN THE ANALYSIS----------------------------------------------------------------------------------------------------------

for conf_min in np.arange(0.4, 1, 0.05):        # Iterate through several confidence thresholds - no point doing 1.0 (as no detections)
    
    conf_min = round(conf_min, 2)

    # New dict to compare species count, above a min conf level - detections in either mono or beamformed
    species_counts = {"mono_channel": {},
                    "beamformed": {}}

    # Analyse all results files in the directory
    for root, dirs, files in os.walk(DIR_PATH, topdown=False):
        for name in files:
            if name == "results.json":
                results_path = os.path.join(root, name)
                current_results_dict = read_results_from_file(results_path)

                species_counts = count_detections_in_dict(current_results_dict, species_counts, conf_min)

    # Setup a new folder for the processed data
    conf_dir_path = DIR_PATH + "/" + "conf_thresholds"

    if not os.path.exists(conf_dir_path):
        os.makedirs(conf_dir_path)

    write_processed_to_file(species_counts, f"conf_thresholds/species_counts_{str(conf_min)}conf.json")

    # Plot the data for visual inspection--------------------------------------
    if conf_min == 0.4 or conf_min == 0.6 or conf_min == 0.8:
        # Species Count Bar Charts...
        mono_dict = species_counts["mono_channel"]
        bf_dict = species_counts["beamformed"]
        barchart_file_path = DIR_PATH + f"/conf_thresholds/bar_compare_count_{str(conf_min)}conf.png"

        barchart_count_comparison(mono_dict, bf_dict, conf_min, barchart_file_path)

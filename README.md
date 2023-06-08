# AMARU Data Analysis
This repository contains source code for analysing 6-channel audio files, captured from the AMARU device.

# Access to Data
Pre-processed data (results of ODAS beamforming & BirdNET predictions) can be found in the data/processed folder.
--> Data regarding confidence level tests is found in 'processed.json' files
--> Data regarding species count tests is found in 'species_counts.json' files
These folders also contain several plots of various findings.

# How to Use
Data analysis scripts are located within the src/analysis_scripts folder.
Use 'process_dir_odas_birdnet.py' to analyse a directory (located in data/raw) of FLAC files -> This will perform ODAS beamforming and provide BirdNET predictions
  --> Results will be published in the data/processed directory, as .json files 
Use 'compare_birdnet_results.py' to analyse .json files and conduct pre-processing of data (as well as plotting several graphs)
Use 'statistical_inference_tests.ipynb' to conduct Wilcoxon Signed-Rank Tests
Use 'species_count_analysis.ipynb' to plot graphs involving species counts from captured data
Use 'SNR_analysis.ipynb' to repeat analysis of the controlled tests, involving the 31-speaker spherical array

***REQUIREMENTS***
To run 'process_dir_odas_birdnet.py', a LINUX laptop is required (owing to the use of ODAS).
ODAS - see the [ODAS GitHub]('https://github.com/introlab/odas') for installation instructions.
birdnetlib, version 0.6.0
Jinja2, version 2.11.3
sox, version 1.4.1

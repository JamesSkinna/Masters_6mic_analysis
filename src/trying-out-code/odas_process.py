from jinja2 import Environment, FileSystemLoader
# Jinja allows us to easily fill-out custom text files (e.g., our config file)
import os
import sox

# Specify which file you want to process...
file_path = "data/IC1/13-05-23/RPiID-000000004e8ff25b/2023-04-18/04-17-09_dur=1200secs.flac"
folder_path = file_path.split(".")[0]

# Setup a new folder for the processed data
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# CONVERT TO RAW------------------------------------------------------------------------
# Input FLAC file
input_file = file_path

# Output RAW audio file
output_file = folder_path + "/raw_6_chan.raw"

# Create a Sox object
tfm = sox.Transformer()

tfm.set_input_format(rate=16000, channels=6, bits=32, encoding="signed-integer")

tfm.set_output_format(rate=16000, channels=6, bits=32, encoding="signed-integer")

# Convert the FLAC file to RAW audio format
tfm.build(input_file, output_file)

# Print a message when the conversion is done
print("Conversion complete!")


# CREATE NEW .cfg FILE------------------------------------------------------------------
# Create a Jinja2 environment with the file system loader
env = Environment(loader=FileSystemLoader("src/configs"))

# Load the template from disk (see the template file for more info - basically, use {{ XXX }} as a placeholder)
template = env.get_template("6mic_post_analysis_template.cfg")

# Setup the paths
raw_input_path = output_file

# Render the template with some data
result = template.render(input_path=raw_input_path, folder_path=folder_path)

# Save the rendered config to file
config_path = folder_path + "/config.cfg"
with open(config_path, "w") as f:
    f.write(result)


# Run ODAS------------------------------------------------------------------------------



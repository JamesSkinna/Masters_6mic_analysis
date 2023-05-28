import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_results_from_file(file_path):
    """Read the BirdNET detections from a file"""

    with open(file_path, 'r') as file:
        results_dict = json.load(file)

    return results_dict


# Where 0 degrees = straight ahead, 90 degrees = directly to the right (clockwise)
# These mappings are based on the directions set in ODAS, and the config of the Speaker Array
# direction_map = {"sep_chan1.mp3": 270,
#                  "sep_chan2.mp3": 180,
#                  "sep_chan3.mp3": 0,
#                  "sep_chan4.mp3": 90}

# Needs investigating - this map seems to make it a lot more accurate (maybe some echoing/deflections inside the device?)
direction_map = {"sep_chan1.mp3": 90,
                 "sep_chan2.mp3": 0,
                 "sep_chan3.mp3": 180,
                 "sep_chan4.mp3": 270}

# Extract the list of directions with highest confidence levels...
processed_file_path = "data/processed/speaker_sphere_lab_tests/15SNR/processed.json"
processed_dict = read_results_from_file(processed_file_path)
max_conf_chan_list = processed_dict["beamformed"]["Eurasian Blackcap"]["max_conf_chan_list"]
start_time_list = processed_dict["beamformed"]["Eurasian Blackcap"]["start_time_list"]

# Map the channel names to the direction
direction_dict = {}
for i, chan in enumerate(max_conf_chan_list):
    direction_dict[int(start_time_list[i])] = direction_map[chan]

# Animate the directions, in the order they were detected------------------------------

# Function to update the plot for each animation frame
def update(frame):
    ax.clear()

    speaker_angle = frame[0]   # Convert the angle to radians
    speaker_dist = frame[1]
    detection_angle = frame[2]
    if speaker_angle != -1:
        speaker_angle = np.deg2rad(speaker_angle)
    if detection_angle != -1:
        detection_angle = np.deg2rad(detection_angle)
    
    # Plot the detection angle, if available
    if detection_angle != -1:
        ax.plot(detection_angle, 0.8, 'o', c='#2C7BB6')
        ax.plot([0, detection_angle], [0, 0.8], 'r--', c='#2C7BB6')
    
    # Plot the speaker angle, if making a noise
    if speaker_angle != -1:
        ax.plot(speaker_angle, speaker_dist, 'o', c='#00FF00', markersize=10)
    
    ax.plot([], [], 'o--', c='#2C7BB6', label='Most confident beamform detection')
    ax.plot([], [], 'o', c='#00FF00', label='Actual direction (position of speaker)')
    angle = np.deg2rad(60)
    ax.legend(loc="lower left",
          bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))
    # plt.legend()

    ax.set_rticks([])
    ax.set_rlim(0, 1.2)
    ax.set_aspect('equal')
    # ax.set_title(f'{frame}°')

    ax.set_theta_direction(-1)  # Set clockwise rotation
    
    # Rotate the entire polar projection
    ax.set_theta_offset(np.pi / 2)  # Rotate by 90 degrees

# Create a figure and axes with polar projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

start_times = np.arange(0, 202, 3)
speaker_directions = []

# Write out where the bird call is actually coming from (at each 3s time interval)
for time in start_times:
    end_time = time + 3
    if end_time < 15:
        # First 15 seconds = speaker 11 -> Direction = 0 degrees (straight ahead)
        # Distance = 1.2 (outer ring of speakers)
        speaker_directions.append([time, 0, 1.2])
    elif end_time > 15 and end_time <= 27:
        # No bird calls
        speaker_directions.append([time, -1, 0])
    elif end_time > 27 and end_time <= 42:
        # Speaker 14 - 108 degrees, outer ring
        speaker_directions.append([time, 108, 1.2])
    elif end_time > 42 and end_time <= 45:
        # No bird calls
        speaker_directions.append([time, -1, 0])
    elif end_time > 45 and end_time <= 63:
        # Speaker 17 - 216 deg, OR
        speaker_directions.append([time, 216, 1.2])
    elif end_time > 63 and end_time <= 66:
        # No bird calls
        speaker_directions.append([time, -1, 0])
    elif end_time > 66 and end_time <= 81:
        # Speaker 20 - 324 deg, OR
        speaker_directions.append([time, 324, 1.2])
    elif end_time > 81 and end_time <= 87:
        # No bird calls
        speaker_directions.append([time, -1, 0])
    elif end_time > 87 and end_time <= 102:
        # Speaker 26 - 342 deg, Inner Ring
        speaker_directions.append([time, 342, 1])
    elif end_time > 102 and end_time <= 108:
        # No bird calls
        speaker_directions.append([time, -1, 0])
    elif end_time > 108 and end_time <= 120:
        # Speaker 27 - 54 deg, Inner Ring
        speaker_directions.append([time, 54, 1])
    elif end_time > 120 and end_time <= 129:
        # No bird calls
        speaker_directions.append([time, -1, 0])
    elif end_time > 129 and end_time <= 141:
        # Speaker 28 - 126 deg, Inner Ring
        speaker_directions.append([time, 126, 1])
    elif end_time > 141 and end_time <= 147:
        # No bird calls
        speaker_directions.append([time, -1, 0])
    elif end_time > 147 and end_time <= 162:
        # Speaker 29 - 198 deg, Inner Ring
        speaker_directions.append([time, 198, 1])
    elif end_time > 162 and end_time <= 168:
        # No bird calls
        speaker_directions.append([time, -1, 0])
    elif end_time > 168 and end_time <= 180:
        # Speaker 30 - 270 deg, Inner Ring
        speaker_directions.append([time, 270, 1])
    elif end_time > 180 and end_time <= 189:
        # No bird calls
        speaker_directions.append([time, -1, 0])
    elif end_time > 189 and end_time <= 201:
        # Speaker 31 - 0 deg (straight up), centre
        speaker_directions.append([time, 0, 0])
    elif end_time > 201 and end_time <= 210:
        # No bird calls
        speaker_directions.append([time, -1, 0])

# print(speaker_directions)
# print(direction_dict)

anim_list = []

for entry in speaker_directions:
    # entry is formatted as a list: [start time, angle, distance]
    start_time = entry[0]
    speaker_angle = entry[1]
    speaker_dist = entry[2]
    if start_time in direction_dict:
        detection_angle = direction_dict[start_time]
    else:
        detection_angle = -1        # Not detected
    anim_list.append([speaker_angle, speaker_dist, detection_angle])

# print(anim_list)

# Create the animation
animation = FuncAnimation(fig, update, frames=anim_list, interval=500, repeat=False)

# Display the animation
plt.show()

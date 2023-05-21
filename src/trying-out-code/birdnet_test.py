from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

recording = Recording(
    analyzer,
    # "data/SpecificTests/OverlappingBirds_Silwood/chan1.mp3",
    "data/SpecificTests/OverlappingBirds_Silwood/6Channels/original_chan1.mp3",
    lat=51.409111,        # Silwood Lat & Long
    lon=-0.637820,
    date=datetime(year=2023, month=5, day=10), # use date or week_48
    min_conf=0.25,
)
recording.analyze()
print(recording.detections)

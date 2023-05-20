from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

recording = Recording(
    analyzer,
    "data/SpecificTests/Controlled_Test/mixkit-melodic-songbird-chirp-in-the-wild-68.wav",
    # lat=51.40644012,        # Silwood Lat & Long
    # lon=-0.63814066,
    date=datetime(year=2023, month=4, day=25), # use date or week_48
    min_conf=0.25,
)
recording.analyze()
print(recording.detections)

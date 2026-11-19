# Wearable Activity Recognition with a Shimmer3 IMU

Senior project at Cal Poly SLO (Jan to Jun 2026) with Christopher Tu. We strapped a Shimmer3 sensor to the chest, streamed the accelerometer data to a laptop over Bluetooth, and trained a Random Forest to tell if someone is resting, walking, or running in real time.

## How it works

1. The Shimmer pairs over Bluetooth and shows up as a COM port. The official Shimmer app wouldn't collect data for us so we talk to it directly over serial at 115200 baud and send the start command (0x07).
2. The Shimmer sends a 117 byte header first, which we find and skip. After that every sample is an 18 byte frame of nine 16 bit values. We only use the accelerometer X, Y, Z.
3. `imu_accel_only.py` records a session to a CSV with a label (rest, walking, running) and shows a live plot while it records.
4. The data gets split into windows of 100 samples (about 2 seconds) with 50% overlap. Each window turns into 15 features: mean, std, min, and max for each axis, plus the mean, std, and range of the overall magnitude.
5. `activity_classifier_live.py` trains a Random Forest (100 trees) on those windows, saves it, and then classifies live data every half second with a dashboard showing the activity and confidence.

The best single feature ended up being the standard deviation of the magnitude. It was around 19 at rest, 78 walking, and 165 running.

## Results

- About 22,000 samples total, roughly 2 minutes per activity per session
- 98% accuracy on a 25% held out test set (110 out of 112 windows)
- Adding jumping as a 4th class dropped it to about 88% since jumping and running look really similar on an accelerometer, so we stuck with 3 classes
- Live it picks up a change in activity within 1 to 2 seconds

## Limitations

The windows overlap, so a random train/test split puts very similar windows on both sides, which probably makes the 98% look better than it would on new data. All the data also came from our own sessions, not new people. Next steps would be splitting by session or by person and adding the gyroscope to help with jumping vs running.

`evaluate.py` is how we got the 98%. It does a random 75/25 split and 5 fold cross validation on the windows. The cross validation averaged about 94% but with a big spread between folds (some were a lot lower), which is another sign the 98% is on the optimistic side.

## Data

The `data/` folder has our actual recordings from April 27, 2026, one session per activity (about 2.5 minutes each at ~51 Hz). The jumping file is from the 4 class experiment and isn't used by the 3 class scripts.

## Running it

```bash
pip install -r requirements.txt
python imu_accel_only.py          # record a labeled session
cd data
python ../evaluate.py                # accuracy, confusion matrix, cross validation
python ../activity_classifier_live.py  # trains on the CSVs in the folder the first time, then runs live
```

Change `PORT` at the top of each script to whatever COM port your Shimmer shows up as.

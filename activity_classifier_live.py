import serial, time, struct, pickle, os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.ensemble import RandomForestClassifier

PORT = "COM3"   # change this to whatever COM port the Shimmer shows up as
BAUD = 115200

FRAME = 18
HEADER = bytes([0xFF, 0xFF, 0x8A, 0x71])
HEADER_LEN = 117

WINDOW_SIZE = 100    # samples per classification window
LABELS = ["Rest", "Walking", "Running"]
COLORS = ["#2ecc71", "#3498db", "#e74c3c"]          # green, blue, red

# ============ TRAIN OR LOAD MODEL ============
MODEL_FILE = "activity_model.pkl"


def extract_features_from_window(window):
    """Extract features from a single window of data"""
    window = np.array(window)
    feat = []

    # Mean of each axis
    feat.extend(window.mean(axis=0))
    # Std of each axis
    feat.extend(window.std(axis=0))
    # Min/Max
    feat.extend(window.min(axis=0))
    feat.extend(window.max(axis=0))

    # Magnitude stats
    mag = np.sqrt((window**2).sum(axis=1))
    feat.append(mag.mean())
    feat.append(mag.std())
    feat.append(mag.max() - mag.min())

    return feat


def train_model():
    """Train model from CSV files in current directory"""
    import pandas as pd

    print("Training model from CSV files...")

    # Find CSV files
    rest_files = [f for f in os.listdir('.') if 'rest' in f.lower() and f.endswith('.csv')]
    walk_files = [f for f in os.listdir('.') if 'walk' in f.lower() and f.endswith('.csv')]
    run_files = [f for f in os.listdir('.') if 'run' in f.lower() and f.endswith('.csv')]

    if not rest_files or not walk_files or not run_files:
        print("ERROR: Need CSV files with 'rest', 'walk', and 'run' in filenames")
        print(f"Found: rest={rest_files}, walk={walk_files}, run={run_files}")
        return None

    def load_and_extract(files, label):
        all_features = []
        for f in files:
            df = pd.read_csv(f)
            data = df[["ax", "ay", "az"]].values
            for start in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE // 2):
                window = data[start:start + WINDOW_SIZE]
                all_features.append(extract_features_from_window(window))
        return np.array(all_features), np.full(len(all_features), label)

    X_rest, y_rest = load_and_extract(rest_files, 0)
    X_walk, y_walk = load_and_extract(walk_files, 1)
    X_run, y_run = load_and_extract(run_files, 2)

    X = np.vstack([X_rest, X_walk, X_run])
    y = np.concatenate([y_rest, y_walk, y_run])

    print(f"Training on {len(X)} windows (Rest: {len(X_rest)}, Walk: {len(X_walk)}, Run: {len(X_run)})")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_FILE}")

    return clf


def load_model():
    """Load existing model or train new one"""
    if os.path.exists(MODEL_FILE):
        print(f"Loading model from {MODEL_FILE}")
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    else:
        return train_model()


# ============ MAIN ============
print("=" * 50)
print("SHIMMER ACTIVITY CLASSIFIER")
print("=" * 50)

# Load or train model
clf = load_model()
if clf is None:
    print("Could not load or train model. Exiting.")
    exit(1)

# Connect to Shimmer
print(f"\nConnecting to Shimmer on {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(0.5)
ser.write(bytes([0x07]))
print("Started streaming.")

# Drop header
buf = bytearray()
t0 = time.time()
while time.time() - t0 < 1.5:
    buf += ser.read(4096)

idx = buf.find(HEADER)
if idx != -1 and idx + HEADER_LEN <= len(buf):
    buf = buf[idx + HEADER_LEN:]
    print("Header dropped.")


def unpack9(frame18):
    return struct.unpack("<9h", frame18)


def frame_looks_bad(v):
    axc, ayc, azc = v[1], v[2], v[3]
    gxc, gyc, gzc = v[4], v[5], v[6]
    bad = sum(abs(x) >= 32760 for x in (axc, ayc, azc, gxc, gyc, gzc))
    return bad >= 2


# ============ SETUP PLOT ============
plt.ion()
fig = plt.figure(figsize=(12, 7))
fig.suptitle("Shimmer Activity Classifier", fontsize=14, fontweight='bold')

# Accelerometer plot (top)
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title("Accelerometer Data")
ax1.set_ylabel("Raw counts")
line_ax, = ax1.plot([], [], 'r-', alpha=0.7, label='X')
line_ay, = ax1.plot([], [], 'g-', alpha=0.7, label='Y')
line_az, = ax1.plot([], [], 'b-', alpha=0.7, label='Z')
ax1.legend(loc='upper right')
ax1.set_xlim(0, 300)
ax1.set_ylim(1500, 2500)

# Classification display (bottom)
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title("Activity Classification")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Big text for current activity
activity_text = ax2.text(0.5, 0.6, "Waiting...", fontsize=48, fontweight='bold',
                         ha='center', va='center', transform=ax2.transAxes)
confidence_text = ax2.text(0.5, 0.25, "", fontsize=20,
                           ha='center', va='center', transform=ax2.transAxes)

# History bar
history_len = 20
classification_history = deque([0] * history_len, maxlen=history_len)

plt.tight_layout()

# ============ DATA BUFFERS ============
window_buffer = deque(maxlen=WINDOW_SIZE)
plot_buffer_ax = deque(maxlen=300)
plot_buffer_ay = deque(maxlen=300)
plot_buffer_az = deque(maxlen=300)

last_plot_time = 0
last_classify_time = 0
start_time = time.time()

print("\nClassifying... Press Ctrl+C to stop.\n")

try:
    while True:
        buf += ser.read(4096)

        while len(buf) >= FRAME:
            frame = bytes(buf[:FRAME])
            v = unpack9(frame)

            if frame_looks_bad(v):
                del buf[0]
                continue

            del buf[:FRAME]

            axc, ayc, azc = v[1], v[2], v[3]

            # Add to buffers
            window_buffer.append([axc, ayc, azc])
            plot_buffer_ax.append(axc)
            plot_buffer_ay.append(ayc)
            plot_buffer_az.append(azc)

        now = time.time()

        # Classify every 0.5 seconds when we have enough data
        if now - last_classify_time > 0.5 and len(window_buffer) >= WINDOW_SIZE:
            last_classify_time = now

            # Extract features and classify
            features = extract_features_from_window(list(window_buffer))
            prediction = clf.predict([features])[0]
            probabilities = clf.predict_proba([features])[0]
            confidence = probabilities[int(prediction)] * 100

            classification_history.append(prediction)

            # Update display
            label = LABELS[int(prediction)]
            color = COLORS[int(prediction)]
            activity_text.set_text(label)
            activity_text.set_color(color)
            confidence_text.set_text(f"Confidence: {confidence:.0f}%")

            # Print to console
            elapsed = int(now - start_time)
            print(f"\r[{elapsed:3d}s] {label:8s} ({confidence:5.1f}%)    |   " +
                  "".join(["▓" if h == prediction else "░" for h in classification_history]), end="")

        # Update plot ~10 fps
        if now - last_plot_time > 0.1 and len(plot_buffer_ax) > 10:
            last_plot_time = now
            n = len(plot_buffer_ax)
            x = np.arange(n)

            line_ax.set_data(x, list(plot_buffer_ax))
            line_ay.set_data(x, list(plot_buffer_ay))
            line_az.set_data(x, list(plot_buffer_az))
            ax1.set_xlim(0, max(n, 10))

            all_vals = list(plot_buffer_ax) + list(plot_buffer_ay) + list(plot_buffer_az)
            if all_vals:
                ymin, ymax = min(all_vals), max(all_vals)
                pad = (ymax - ymin) * 0.1 if ymax > ymin else 100
                ax1.set_ylim(ymin - pad, ymax + pad)

            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\n\nStopping...")

finally:
    ser.close()
    print("Done.")

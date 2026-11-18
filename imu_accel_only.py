import serial, time, struct, csv
import numpy as np
import matplotlib.pyplot as plt

PORT = "COM3"   # change this to whatever COM port the Shimmer shows up as
BAUD = 115200

FRAME = 18
HEADER = bytes([0xFF, 0xFF, 0x8A, 0x71])
HEADER_LEN = 117

# Connect
ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(0.5)
ser.write(bytes([0x07]))        # start streaming
print("Started streaming.")

# Grab initial bytes and drop header
buf = bytearray()
t0 = time.time()
while time.time() - t0 < 1.5:
    buf += ser.read(4096)

idx = buf.find(HEADER)
if idx != -1 and idx + HEADER_LEN <= len(buf):
    buf = buf[idx + HEADER_LEN:]
    print("Dropped header.")
else:
    print("Header not found cleanly; continuing anyway.")


def unpack9(frame18: bytes):
    return struct.unpack("<9h", frame18)


# CSV logging - prompts for activity label
print("\n=== DATA COLLECTION ===")
print("Activities: rest, walking, running, jumping")
label = input("Enter activity label for this session: ").strip().lower()

ts = time.strftime("%Y%m%d_%H%M%S")
csv_path = f"shimmer_{label}_{ts}.csv"
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["t", "ax", "ay", "az", "accel_mag", "label"])

print(f"\nLogging to: {csv_path}")
print("Recording... Press Ctrl+C to stop.\n")

# Live plot - ONLY accelerometer
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Top plot: individual axes
ax1.set_title(f"Accelerometer XYZ - {label}")
ax1.set_xlabel("Samples")
ax1.set_ylabel("Raw counts")
line_ax, = ax1.plot([], [], 'r-', label='ax')
line_ay, = ax1.plot([], [], 'g-', label='ay')
line_az, = ax1.plot([], [], 'b-', label='az')
ax1.legend(loc='upper right')

# Bottom plot: magnitude
ax2.set_title("Accelerometer Magnitude")
ax2.set_xlabel("Samples")
ax2.set_ylabel("Magnitude")
line_mag, = ax2.plot([], [], 'purple')

window = 300
data_ax = []
data_ay = []
data_az = []
data_mag = []

last_plot = 0.0
start_time = time.time()
sample_count = 0


def frame_looks_bad(v):
    axc, ayc, azc = v[1], v[2], v[3]
    gxc, gyc, gzc = v[4], v[5], v[6]
    bad = sum(abs(x) >= 32760 for x in (axc, ayc, azc, gxc, gyc, gzc))
    return bad >= 2


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
            mag = (axc**2 + ayc**2 + azc**2) ** 0.5

            t = time.time() - start_time
            sample_count += 1

            # Log to CSV
            writer.writerow([f"{t:.3f}", axc, ayc, azc, f"{mag:.1f}", label])

            # Store for plotting
            data_ax.append(axc)
            data_ay.append(ayc)
            data_az.append(azc)
            data_mag.append(mag)

            if len(data_ax) > window:
                data_ax.pop(0)
                data_ay.pop(0)
                data_az.pop(0)
                data_mag.pop(0)

        # Update plot ~10 fps
        now = time.time()
        if now - last_plot > 0.10 and len(data_ax) > 5:
            last_plot = now
            n = len(data_ax)
            x = np.arange(n)

            # Update XYZ plot
            line_ax.set_data(x, data_ax)
            line_ay.set_data(x, data_ay)
            line_az.set_data(x, data_az)

            all_vals = data_ax + data_ay + data_az
            ymin, ymax = min(all_vals), max(all_vals)
            pad = (ymax - ymin) * 0.1 if ymax > ymin else 100
            ax1.set_xlim(0, max(10, n))
            ax1.set_ylim(ymin - pad, ymax + pad)

            # Update magnitude plot
            line_mag.set_data(x, data_mag)
            ax2.set_xlim(0, max(10, n))
            mag_min, mag_max = min(data_mag), max(data_mag)
            pad2 = (mag_max - mag_min) * 0.1 if mag_max > mag_min else 100
            ax2.set_ylim(mag_min - pad2, mag_max + pad2)

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Print status
            elapsed = int(now - start_time)
            print(f"\rRecording: {elapsed}s, {sample_count} samples", end="")

except KeyboardInterrupt:
    print("\n\nStopping...")

finally:
    csv_file.close()
    ser.close()
    duration = time.time() - start_time
    print(f"Saved: {csv_path}")
    print(f"Duration: {duration:.1f}s, Samples: {sample_count}")

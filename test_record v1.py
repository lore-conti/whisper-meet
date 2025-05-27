"""
# Audio Recording Test Script v1

Simple test script to record audio using sounddevice and save as WAV file.
Includes interactive countdown and early stop support.

## Features
- Records mono audio at CD quality (44.1kHz)
- Shows interactive countdown during recording
- Allows early stopping with Ctrl+C
- Saves WAV files with timestamps
- Organized file storage in 'recordings' folder

## Usage
```powershell
python test_record_v1.py
```

## Output
Saves recording as WAV file:
recordings/record_YYYYMMDD_HHMMSS.wav
"""

import os
import sys
import time
import datetime
import sounddevice as sd
from scipy.io.wavfile import write

# === Configuration ===
SAMPLE_RATE = 44100  # CD quality
DURATION = 5         # seconds
CHANNELS = 1         # mono
RECORDINGS_DIR = "recordings"  # recordings folder name

# Create recordings directory if it doesn't exist
try:
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    print(f"[INFO] Using recordings directory: {os.path.abspath(RECORDINGS_DIR)}")
except Exception as e:
    print(f"[ERROR] Failed to create recordings directory: {e}")
    sys.exit(1)

print(sd.query_devices())

sd.default.device = 1

# === Prepare filename ===
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(RECORDINGS_DIR, f"record_{now}.wav")
print(f"[DEBUG] Output audio filename: {filename}")


# === Record Audio with interactive countdown ===
try:
    duration = DURATION
    print(f"[INFO] Recording for {DURATION} seconds... Press Ctrl+C to stop early.")
    start_time = time.time()
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    while True:
        elapsed = int(time.time() - start_time)
        remaining = int(duration - elapsed)
        if remaining <= 0:
            break
        print(f"\r[RECORDING] {elapsed}s elapsed, {remaining}s remaining", end="")
        time.sleep(1)
    sd.wait()
    print("\n[INFO] Recording complete.")
except KeyboardInterrupt:
    print("\n[WARNING] Recording manually interrupted! Saving partial audio...")
    sd.stop()
except Exception as e:
    print(f"[ERROR] Recording failed: {e}")
    sys.exit(1)

# === Save WAV file ===
try:
    write(filename, SAMPLE_RATE, audio)
    print(f"[INFO] Audio saved to {filename}")
    print(f"[DEBUG] Full path: {os.path.abspath(filename)}")
except Exception as e:
    print(f"[ERROR] Failed to write WAV file: {e}")
    sys.exit(1)
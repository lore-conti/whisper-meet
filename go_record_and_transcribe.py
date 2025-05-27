"""
WhisperMeet Recording and Transcription Script

This script records audio from system audio and microphone, then transcribes it using WhisperX.
Features:
- Records audio for a specified duration (default: 60 minutes)
- Uses WhisperX for accurate transcription
- Performs speaker diarization to identify different speakers
- Aligns audio with transcription for accurate timestamps
- Outputs a formatted Markdown file with timestamped speaker segments
- Supports early stopping with Ctrl+C
"""

import whisperx
import datetime
import sounddevice as sd
from scipy.io.wavfile import write
import os
import sys
import time
from whisperx.diarize import DiarizationPipeline
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

if not HF_TOKEN:
    raise ValueError("[ERROR] HF_TOKEN not found in .env file. Please add your Hugging Face token.")

# === Configuration ===
DURATION_MINUTES = 60
SAMPLERATE = 16000
CHANNELS = 1
MODEL_SIZE = "medium"
DEVICE = "cpu"

print("[DEBUG] Script started.")
print(f"[DEBUG] Current working directory: {os.getcwd()}")
print(f"[DEBUG] Using Whisper model: {MODEL_SIZE} on device: {DEVICE}")
print(f"[DEBUG] Audio recording duration: {DURATION_MINUTES} minutes")

# === Prepare filename ===
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"meeting_{now}.wav"
print(f"[DEBUG] Output audio filename: {filename}")

# === Record Audio with interactive countdown ===
try:
    duration = DURATION_MINUTES * 60
    print(f"[INFO] Recording for {DURATION_MINUTES} minutes... Press Ctrl+C to stop early.")
    start_time = time.time()
    audio = sd.rec(int(duration * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS)
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
    write(filename, SAMPLERATE, audio)
    print(f"[INFO] Audio saved to {filename}")
except Exception as e:
    print(f"[ERROR] Failed to write WAV file: {e}")
    sys.exit(1)

# === Load WhisperX model and transcribe (float32 for CPU) ===
try:
    print("[INFO] Loading WhisperX model...")
    model = whisperx.load_model(MODEL_SIZE, device=DEVICE, compute_type="float32")
    print("[INFO] Transcribing audio...")
    result = model.transcribe(filename)
    print("[DEBUG] Transcription result (first 3 segments):")
    for seg in result['segments'][:3]:
        print(f"  -> {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")
except Exception as e:
    print(f"[ERROR] Transcription failed: {e}")
    sys.exit(1)

# === Alignment ===
try:
    print("[INFO] Performing alignment...")
    align_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    result = whisperx.align(result["segments"], align_model, metadata, filename, DEVICE)
    print("[DEBUG] Alignment successful. Sample word segment:")
    print(result["word_segments"][0])
except Exception as e:
    print(f"[ERROR] Alignment failed: {e}")
    sys.exit(1)

# === Diarization ===
try:
    print("[INFO] Running diarization...")
    diarization_pipeline = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    diarization_segments = diarization_pipeline(filename)
    print(f"[DEBUG] Diarization complete. Found {len(diarization_segments)} speaker segments.")
except Exception as e:
    print(f"[ERROR] Diarization failed: {e}")
    sys.exit(1)

# === Assign speakers ===
try:
    print("[INFO] Assigning speaker labels...")
    if not result.get("word_segments"):
        raise ValueError("No word_segments found in aligned transcription")

    # Debug the word segments structure
    print("[DEBUG] Sample word segment structure:", result["word_segments"][0] if result["word_segments"] else "None")

    # Assign speakers and store the result
    result = whisperx.assign_word_speakers(diarization_segments, result)

    if not isinstance(result["word_segments"], list):
        raise TypeError("Expected word_segments to be a list")

    # Group words by speaker
    grouped = defaultdict(list)
    for word in result["word_segments"]:
        if isinstance(word, dict):
            speaker = word.get("speaker", "SPEAKER_UNKNOWN")
            # Use 'word' key instead of 'text' if that's how it's structured
            word_text = word.get("word", "") or word.get("text", "")
            if word_text:  # Only append if we have text
                grouped[speaker].append({
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "text": word_text
                })

    # Create final segments from grouped words
    final_segments = []
    for speaker, words in grouped.items():
        if words:  # Check if there are any words for this speaker
            start = words[0]["start"]
            end = words[-1]["end"]
            text = " ".join(w["text"] for w in words)
            final_segments.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text
            })

    if final_segments:
        print("[DEBUG] First speaker-labeled segment:")
        print(final_segments[0])
    else:
        print("[DEBUG] Word segments structure:", result["word_segments"])
        raise ValueError("No speaker segments were created")

except Exception as e:
    print(f"[ERROR] Speaker assignment failed: {e}")
    print(f"[DEBUG] Result type: {type(result)}")
    print(f"[DEBUG] Word segments sample:", 
          result["word_segments"][0] if result.get("word_segments") else "None")
    sys.exit(1)

# === Save Markdown output ===
try:
    output_md = filename.replace(".wav", ".md")
    print(f"[INFO] Writing Markdown transcript to {output_md}...")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# Transcript – {now}\n\n")
        for segment in final_segments:
            speaker = segment.get("speaker", "Speaker ?")
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            f.write(f"### {speaker} ({start:.2f}s – {end:.2f}s)\n\n{text}\n\n")
    print("[SUCCESS] Markdown transcript saved.")
except Exception as e:
    print(f"[ERROR] Failed to save markdown file: {e}")
    sys.exit(1)

print("[DONE] All steps completed successfully.")


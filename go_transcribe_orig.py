"""
# WhisperX Transcription with Speaker Diarization

This script transcribes WAV audio files using WhisperX and performs speaker diarization.

## Features
- Transcribes audio using WhisperX's medium model
- Performs speaker diarization to identify different speakers
- Aligns audio with transcription for accurate timestamps
- Outputs a formatted Markdown file with timestamped speaker segments
- Creates transcript in the same directory as input file

## Usage
```
python go_transcribe_orig.py input_file.wav
```

## Output Format
Creates a Markdown file with:
- Timestamp of transcription
- Source file information
- Speaker-labeled segments with timestamps
- Cleaned and formatted text
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
import argparse

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

if not HF_TOKEN:
    raise ValueError("[ERROR] HF_TOKEN not found in .env file. Please add your Hugging Face token.")

# === Configuration ===
MODEL_SIZE = "medium"
DEVICE = "cpu"

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Transcribe WAV file with speaker diarization'
    )
    parser.add_argument(
        'input_file',
        help='Path to the WAV file to transcribe'
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"[ERROR] File not found: {args.input_file}")
        sys.exit(1)
    if not args.input_file.endswith('.wav'):
        print(f"[ERROR] File must be a WAV file: {args.input_file}")
        sys.exit(1)

    print("[DEBUG] Script started.")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Input file: {args.input_file}")
    print(f"[DEBUG] Using Whisper model: {MODEL_SIZE} on device: {DEVICE}")

    # Replace the filename variable with args.input_file
    filename = args.input_file

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
        # Initialize pipeline
        diarization_pipeline = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarization_segments = diarization_pipeline(filename)
        print(f"[DEBUG] Diarization complete. Found {len(diarization_segments)} speaker segments.")
    except Exception as e:
        print(f"[ERROR] Diarization failed: {e}")
        print("[DEBUG] Full error:", str(e))
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
        # Get directory and base name of input file
        input_dir = os.path.dirname(os.path.abspath(args.input_file))
        input_base = os.path.basename(args.input_file)
        
        # Create output filename in same directory
        output_md = os.path.join(input_dir, input_base.replace(".wav", ".md"))
        
        print(f"[INFO] Writing Markdown transcript to {output_md}...")
        
        with open(output_md, "w", encoding="utf-8") as f:
            # Add input file information to transcript
            f.write(f"# Transcript – {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"Source: {input_base}\n\n")
            
            for segment in final_segments:
                speaker = segment.get("speaker", "Speaker ?")
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                f.write(f"### {speaker} ({start:.2f}s – {end:.2f}s)\n\n{text}\n\n")
        
        print(f"[SUCCESS] Markdown transcript saved to: {output_md}")
        print(f"[DEBUG] Transcript directory: {input_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to save markdown file: {e}")
        print(f"[DEBUG] Attempted output path: {output_md}")
        sys.exit(1)

    print("[DONE] All steps completed successfully.")

if __name__ == "__main__":
    main()


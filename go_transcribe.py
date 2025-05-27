"""
# WhisperX Audio Transcription with Speaker Diarization

A script that transcribes WAV audio files using WhisperX and performs speaker diarization 
with enhanced confidence scoring and segment merging.

## Features
- Transcribes audio using WhisperX's medium model
- Performs speaker diarization and alignment
- Merges close segments from the same speaker
- Calculates speaker confidence scores
- Outputs a formatted Markdown file with:
  - Speaker statistics (talk time, segments, confidence)
  - Timestamped transcriptions
  - Speaker-labeled segments

## Usage
```powershell
python go_transcribe.py path/to/audio.wav
```

## Output
Creates a Markdown file next to the input file containing:
- Speaker statistics and confidence scores
- Timestamped transcript with speaker labels
- Merged and cleaned text segments
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

# Diarization configuration
DIARIZATION_CONFIG = {
    'min_speakers': 2,
    'max_speakers': 5,
    'min_duration': 0.2,     # Minimum duration for speech segments
    'threshold': 0.4,        # Threshold for voice activity detection
    'uri_key': None          # For consistent speaker labeling across files
}

def merge_close_segments(segments, max_gap=1.0):
    """Merge segments from the same speaker that are close together"""
    merged = []
    current = None
    
    for segment in sorted(segments, key=lambda x: x['start']):
        if not current:
            current = segment.copy()
            continue
            
        if (segment['start'] - current['end'] <= max_gap and 
            segment['speaker'] == current['speaker']):
            # Merge segments
            current['end'] = segment['end']
            current['text'] += ' ' + segment['text']
        else:
            merged.append(current)
            current = segment.copy()
    
    if current:
        merged.append(current)
    
    return merged

def calculate_speaker_confidence(segments):
    """Calculate confidence scores for speaker assignments"""
    speaker_stats = defaultdict(lambda: {'duration': 0, 'segments': 0})
    
    for segment in segments:
        speaker = segment['speaker']
        duration = segment['end'] - segment['start']
        speaker_stats[speaker]['duration'] += duration
        speaker_stats[speaker]['segments'] += 1
    
    total_duration = sum(s['duration'] for s in speaker_stats.values())
    
    for speaker in speaker_stats:
        stats = speaker_stats[speaker]
        stats['confidence'] = (stats['duration'] / total_duration + 
                             stats['segments'] / len(segments)) / 2
    
    return speaker_stats

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
        # Initialize pipeline without speaker parameters
        diarization_pipeline = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)

        # Run diarization with speaker parameters
        diarization_segments = diarization_pipeline(
            filename,
            num_speakers=DIARIZATION_CONFIG['max_speakers'],  # Use max_speakers as num_speakers
            min_speakers=DIARIZATION_CONFIG['min_speakers'],
            max_speakers=DIARIZATION_CONFIG['max_speakers']
        )
        
        print(f"[DEBUG] Diarization complete. Found {len(diarization_segments)} speaker segments.")
        print(f"[DEBUG] First diarization segment: {next(iter(diarization_segments))}")

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

        # Merge close segments from the same speaker
        final_segments = merge_close_segments(final_segments)

        # Calculate speaker statistics
        speaker_stats = calculate_speaker_confidence(final_segments)
        print("\n[INFO] Speaker Statistics:")
        for speaker, stats in speaker_stats.items():
            print(f"  {speaker}:")
            print(f"    - Talk time: {stats['duration']:.1f}s")
            print(f"    - Segments: {stats['segments']}")
            print(f"    - Confidence: {stats['confidence']:.2%}")

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

    # === Save Markdown output with confidence scores ===
    try:
        output_md = filename.replace(".wav", ".md")
        print(f"[INFO] Writing Markdown transcript to {output_md}...")
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(f"# Transcript – {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\n\n")
            
            # Add speaker statistics to the transcript
            f.write("## Speaker Statistics\n\n")
            for speaker, stats in speaker_stats.items():
                f.write(f"- {speaker}:\n")
                f.write(f"  - Talk time: {stats['duration']:.1f}s\n")
                f.write(f"  - Segments: {stats['segments']}\n")
                f.write(f"  - Confidence: {stats['confidence']:.2%}\n\n")
            
            f.write("## Transcript\n\n")
            for segment in final_segments:
                speaker = segment.get("speaker", "Speaker ?")
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                confidence = speaker_stats[speaker]['confidence']
                f.write(f"### {speaker} ({start:.2f}s – {end:.2f}s) [Confidence: {confidence:.2%}]\n\n{text}\n\n")
        print("[SUCCESS] Markdown transcript saved.")
    except Exception as e:
        print(f"[ERROR] Failed to save markdown file: {e}")
        sys.exit(1)

    print("[DONE] All steps completed successfully.")

if __name__ == "__main__":
    main()


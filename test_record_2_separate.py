"""
# Meeting Audio Recorder

Records both system audio and microphone input simultaneously in separate files.
Provides a command-line interface for controlling recordings.

## Features
- Records microphone input (mono, 16kHz)
- Records system audio output (stereo, 16kHz)
- Saves separate WAV files for each audio source
- Supports manual and timed recordings
- Lists available audio devices
- Organizes recordings in timestamped files

## Usage
```powershell
python test_record_2_separate.py
```

## Output Files
Saves in 'recordings' directory:
- mic_YYYYMMDD_HHMMSS.wav (microphone audio)
- system_YYYYMMDD_HHMMSS.wav (system audio)
"""

import pyaudio
import wave
import threading
import time
import os
from datetime import datetime
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

class MeetingRecorder:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1 
        self.rate = 16000
        self.recording = False
        self.frames_mic = []
        self.frames_system = []
        self.recordings_dir = "recordings"
        
        # Create recordings directory if it doesn't exist
        try:
            os.makedirs(self.recordings_dir, exist_ok=True)
            print(f"[INFO] Using recordings directory: {os.path.abspath(self.recordings_dir)}")
        except Exception as e:
            print(f"[ERROR] Failed to create recordings directory: {e}")
            raise
    
    def list_audio_devices(self):
        """List all available audio devices"""
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} - {device['max_input_channels']} in, {device['max_output_channels']} out")
        return devices
    
    def record_microphone(self, filename):
        """Record from microphone"""
        audio = pyaudio.PyAudio()
        
        # Find default input device
        try:
            stream = audio.open(
                format=self.format,
                channels=1,  # Mono for microphone
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            print("Recording microphone...")
            while self.recording:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.frames_mic.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # Save microphone recording
            wf = wave.open(filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames_mic))
            wf.close()
            
        except Exception as e:
            print(f"Error recording microphone: {e}")
        finally:
            audio.terminate()
    
    def record_system_audio(self, filename):
        """Record system audio using sounddevice (works better for system audio)"""
        try:
            print("Recording system audio...")
            
            # Get system audio (this requires specific setup depending on OS)
            duration_chunk = 0.1  # Record in small chunks
            system_audio = []
            
            while self.recording:
                # Record system audio - this may need adjustment based on your system
                try:
                    # Use loopback device if available (Windows) or soundflower (Mac)
                    audio_chunk = sd.rec(
                        int(self.rate * duration_chunk), 
                        samplerate=self.rate, 
                        channels=2,
                        device=None  # Use default output device
                    )
                    sd.wait()
                    system_audio.append(audio_chunk)
                except Exception as e:
                    print(f"System audio recording error: {e}")
                    time.sleep(duration_chunk)
            
            if system_audio:
                # Combine all chunks
                full_audio = np.concatenate(system_audio, axis=0)
                wavfile.write(filename, self.rate, full_audio)
                
        except Exception as e:
            print(f"Error recording system audio: {e}")
    
    def get_recording_path(self, prefix="meeting"):
        """Generate a timestamped filepath in the recordings directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.wav"
        return os.path.join(self.recordings_dir, filename)
    
    def start_recording(self, output_dir=None):
        """Start recording both microphone and system audio"""
        if self.recording:
            print("[WARNING] Already recording!")
            return
        
        # Use default recordings directory if none specified
        output_dir = output_dir or self.recordings_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filenames with timestamp
        mic_filename = self.get_recording_path(prefix="mic")
        system_filename = self.get_recording_path(prefix="system")
        
        print(f"[INFO] Starting recording...")
        print(f"[DEBUG] Microphone recording path: {os.path.abspath(mic_filename)}")
        print(f"[DEBUG] System audio recording path: {os.path.abspath(system_filename)}")
        
        self.recording = True
        self.frames_mic = []
        self.frames_system = []
        
        # Start recording threads
        mic_thread = threading.Thread(target=self.record_microphone, args=(mic_filename,))
        system_thread = threading.Thread(target=self.record_system_audio, args=(system_filename,))
        
        mic_thread.start()
        system_thread.start()
        
        return [mic_thread, system_thread]
    
    def stop_recording(self):
        """Stop recording"""
        if not self.recording:
            print("Not currently recording!")
            return
        
        print("Stopping recording...")
        self.recording = False
    
    def record_for_duration(self, duration_minutes, output_dir="recordings"):
        """Record for a specific duration"""
        threads = self.start_recording(output_dir)
        if threads:
            print(f"Recording for {duration_minutes} minutes...")
            time.sleep(duration_minutes * 60)
            self.stop_recording()
            
            # Wait for threads to finish
            for thread in threads:
                thread.join()
            
            print("Recording completed!")

def main():
    recorder = MeetingRecorder()
    
    print("Meeting Audio Recorder")
    print("=====================")
    
    # List available devices
    recorder.list_audio_devices()
    
    while True:
        print("\nOptions:")
        print("1. Start recording")
        print("2. Stop recording")
        print("3. Record for specific duration")
        print("4. List audio devices")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            try:
                recorder.start_recording()
                print("Recording started. Press Enter to stop or choose option 2.")
            except Exception as e:
                print(f"Error starting recording: {e}")
        
        elif choice == '2':
            recorder.stop_recording()
        
        elif choice == '3':
            try:
                duration = float(input("Enter duration in minutes: "))
                recorder.record_for_duration(duration)
            except ValueError:
                print("Please enter a valid number for duration.")
            except Exception as e:
                print(f"Error during timed recording: {e}")
        
        elif choice == '4':
            recorder.list_audio_devices()
        
        elif choice == '5':
            if recorder.recording:
                recorder.stop_recording()
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
"""
# Meeting Audio Recorder with Combined Output

A script that simultaneously records system audio and microphone input, 
mixing them into a single WAV file. Designed for recording online meetings 
with both local speaker and remote participant audio.

## Features
- Records and mixes microphone and system audio
- Saves as 16kHz mono WAV file (optimized for transcription)
- Interactive command-line interface
- Supports manual and timed recordings
- Lists available audio devices
- Organized file storage in 'recordings' folder

## Usage
```powershell
python test_record_merge.py
```

## Output Files
Saves in 'recordings' directory:
meeting_combined_YYYYMMDD_HHMMSS.wav (mixed audio)
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
        self.channels = 1  # Mono for all audio
        self.rate = 16000  # Changed from 441000 to standard 16kHz
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
    
    def get_recording_path(self, prefix="meeting"):
        """Generate a timestamped filepath in the recordings directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.wav"
        return os.path.join(self.recordings_dir, filename)
        
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
    
    def record_combined_audio(self, filename):
        """Record both microphone and system audio into a single file"""
        try:
            print("Recording combined audio...")
            
            # Initialize audio streams
            audio = pyaudio.PyAudio()
            
            # Calculate buffer sizes
            duration_chunk = 0.1  # 100ms chunks
            samples_per_chunk = int(self.rate * duration_chunk)
            
            mic_stream = audio.open(
                format=self.format,
                channels=self.channels,  # Mono
                rate=self.rate,
                input=True,
                frames_per_buffer=samples_per_chunk
            )
            
            # Initialize numpy arrays for combined audio
            combined_audio = []
            
            while self.recording:
                # Record system audio chunk first
                sys_data = sd.rec(
                    samples_per_chunk,
                    samplerate=self.rate,
                    channels=self.channels,  # Mono
                    dtype=np.int16,
                    device=None
                )
                sd.wait()
                
                # Record microphone chunk of the same size
                mic_data = np.frombuffer(
                    mic_stream.read(samples_per_chunk, exception_on_overflow=False),
                    dtype=np.int16
                )
                
                # Both arrays are now mono, no need for reshaping
                # Mix microphone and system audio with equal weights
                mixed_data = np.clip(mic_data + sys_data.flatten(), -32768, 32767).astype(np.int16)
                combined_audio.append(mixed_data)
            
            # Cleanup streams
            mic_stream.stop_stream()
            mic_stream.close()
            audio.terminate()
            
            if combined_audio:
                # Combine all chunks and save
                full_audio = np.concatenate(combined_audio)
                wavfile.write(filename, self.rate, full_audio)
                print(f"Saved combined audio to: {filename}")
                
        except Exception as e:
            print(f"Error recording combined audio: {e}")
            print(f"Debug info - mic shape: {mic_data.shape if 'mic_data' in locals() else 'N/A'}")
            print(f"Debug info - sys shape: {sys_data.shape if 'sys_data' in locals() else 'N/A'}")

    def start_recording(self, output_dir=None):
        """Start recording combined audio"""
        if self.recording:
            print("[WARNING] Already recording!")
            return
        
        # Use default recordings directory if none specified
        output_dir = output_dir or self.recordings_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        combined_filename = self.get_recording_path(prefix="meeting_combined")
        
        print(f"[INFO] Starting recording...")
        print(f"[DEBUG] Audio will be saved to: {os.path.abspath(combined_filename)}")
        
        self.recording = True
        
        # Start recording thread
        record_thread = threading.Thread(
            target=self.record_combined_audio,
            args=(combined_filename,)
        )
        record_thread.start()
        
        return [record_thread]
    
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
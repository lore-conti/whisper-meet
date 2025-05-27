# WhisperX Meeting Transcription with Speaker Diarization

Record and transcribe meetings with speaker detection, outputting to Markdown — fully offline on Windows.

## ✨ Features

- Record system audio and microphone simultaneously
- Transcribe audio using WhisperX
- Detect and label different speakers
- Generate timestamped Markdown transcripts
- Calculate speaker confidence scores
- Support for manual and timed recordings

## 📋 Requirements

- Windows 10 or 11 (64-bit)
- Python 3.9+
- Git
- FFmpeg
- Hugging Face account

## 🚀 Getting Started

### 1. Install Required Software

1. **Python**: Download and install from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation
   - Verify: `python --version`

2. **Git**: Install from [git-scm.com](https://git-scm.com/downloads)
   - Verify: `git --version`

3. **FFmpeg**: Install using one of these methods:
   ```powershell
   # Option 1: Using Chocolatey (recommended)
   choco install ffmpeg

   # Option 2: Manual installation
   # Download from ffmpeg.org and add to PATH
   ```
   - Verify: `ffmpeg -version`

### 2. Project Setup

1. **Create Project**:
   ```powershell
   mkdir C:\WhisperMeet
   cd C:\WhisperMeet
   python -m venv whisperx-env
   whisperx-env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Hugging Face**:
   - Create account at [huggingface.co](https://huggingface.co)
   - Accept terms for required models:
     - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
     - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
   - Get token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Update `.env` file:
     ```plaintext
     HF_TOKEN=your_token_here
     ```

## 📝 Usage

### Recording Meetings

1. **Start Recording**:
   ```powershell
   cd C:\WhisperMeet
   whisperx-env\Scripts\activate
   python test_record_merge.py
   ```

2. **Available Commands**:
   - `1` - Start recording
   - `2` - Stop recording
   - `3` - Record for specific duration
   - `4` - List audio devices
   - `5` - Exit

### Transcribing Recordings

```powershell
python go_transcribe.py recordings\your_recording.wav
```

Output will be saved as a Markdown file next to the input:
```markdown
# Transcript – 20250525_153010

### Speaker 0 (0.00s – 4.21s) [Confidence: 95%]
Welcome to the meeting. Let's begin.

### Speaker 1 (4.21s – 6.88s) [Confidence: 92%]
Thanks! I'll start with the updates from last week.
```

## 🔧 Troubleshooting

- **Audio Issues**: Run as Administrator
- **Installation Errors**: Check Python PATH
- **SSL Errors**: Use trusted hosts:
  ```powershell
  pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
  ```

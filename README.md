---
title: Piano MIDI Transcription
sdk: gradio
app_file: app.py
pinned: false
---

# Piano → MIDI Transcription

Upload a piano recording and get a downloadable MIDI file. Works with mp3, wav, m4a, flac, and ogg formats.

## Usage

1. Upload an audio file containing piano music
2. Click "Transcribe" 
3. Download the generated MIDI file

## Tips

- Shorter clips (1-3 minutes) are much faster
- First run will download model weights (may take a moment)
- Works best with clear piano recordings

## Technical Details

- Uses the `piano_transcription_inference` library
- Model: Note F1=0.9677, Pedal F1=0.9186
- Supports various audio formats via ffmpeg

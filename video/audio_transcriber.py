import whisper
import json
import os

def transcribe_audio(audio_path, output_dir):
    """
    Transcribe audio using OpenAI's Whisper model and save results with timestamps.
    
    Args:
        audio_path (str): Path to the audio file
        output_path (str): Path to save the JSON output
    
    Returns:
        list: List of dictionaries containing timestamps and transcribed text
    """
    print("Loading Whisper model...")
    model = whisper.load_model("tiny")
    
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    
    # Format the segments into the requested format
    segments = []
    for segment in result["segments"]:
        segments.append({
            "timestamp": segment["start"],
            "text": segment["text"].strip()
        })
    
    # Save to JSON file
    audio_transcription_path = os.path.join(output_dir, "audio_transcription.json")
    with open(audio_transcription_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    
    return segments 
import whisper
import json
import os

def transcribe_audio(audio_path, output_dir):
    """
    Transcribe audio using OpenAI's Whisper model and save results with timestamps.
    """
    print("Loading Whisper model...")
    model = whisper.load_model("tiny")

    # Normalize the path
    audio_path = os.path.abspath(audio_path)
    print(f"\n\nAudio Path: {audio_path}")

    # Check if file exists
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file does not exist at {audio_path}")
        return []

    try:
        result = model.transcribe(audio_path)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return []

    print("Transcription completed....")

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
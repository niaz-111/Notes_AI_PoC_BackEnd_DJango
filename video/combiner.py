import json
import os

def combine_transcriptions(audio_transcription, frame_descriptions, output_dir):
    """
    Combine audio transcription with frame descriptions based on timestamps.
    Distributes frames to their nearest audio segments.
    
    Args:
        audio_transcription (list): List of audio segments with timestamps
        frame_descriptions (list): List of frame descriptions with timestamps
        output_dir (str): Directory to save the combined output
        
    Returns:
        str: Path to the saved combined file
    """
    combined_data = []
    
    # Sort both lists by timestamp for efficient matching
    audio_segments = sorted(audio_transcription, key=lambda x: x["timestamp"])
    frames = sorted(frame_descriptions, key=lambda x: x["timestamp"])
    
    # Create a mapping of frames to their nearest audio segments
    frame_to_audio = {}
    for frame in frames:
        # Find the audio segment with the closest timestamp
        closest_audio = min(audio_segments, key=lambda x: abs(x["timestamp"] - frame["timestamp"]))
        audio_time = closest_audio["timestamp"]
        
        # Initialize the list for this audio segment if it doesn't exist
        if audio_time not in frame_to_audio:
            frame_to_audio[audio_time] = []
        
        # Add the frame to the list for this audio segment
        frame_to_audio[audio_time].append({
            "description": frame["text"],
            "timestamp": frame["timestamp"]
        })
    
    # Create combined entries for each audio segment
    for audio_segment in audio_segments:
        audio_time = audio_segment["timestamp"]
        
        # Create the base entry with audio information
        combined_entry = {
            "timestamp": audio_time,
            "audio_text": audio_segment["text"]
        }
        
        # Add frame information if available for this audio segment
        if audio_time in frame_to_audio:
            frames_for_segment = frame_to_audio[audio_time]
            if frames_for_segment:
                # Add all frames for this segment
                combined_entry["frames"] = frames_for_segment
        
        combined_data.append(combined_entry)
    
    # Save combined data to JSON file
    output_path = os.path.join(output_dir, "combined_transcription.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    return output_path 
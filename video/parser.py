from .audio_extractor import extract_audio
from .frame_extractor import extract_good_frames
import os
import shutil

def process_video(video_path, output_dir):
    # Delete output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "audio.wav")
    frames_dir = os.path.join(output_dir, "frames")

    print("\naudio_path:", audio_path)

    audio = extract_audio(video_path, audio_path)
    frames = extract_good_frames(video_path, frames_dir)

    return {
        "audio": audio_path,
        "frames": frames
    }


from moviepy.editor import VideoFileClip
from tqdm import tqdm

def extract_audio(video_path, audio_path):
    print("Extracting audio...")
    clip = VideoFileClip(video_path)
    with tqdm(total=100, desc="Audio extraction") as pbar:
        clip.audio.write_audiofile(audio_path, logger=None)
        pbar.update(100)
    clip.close()
    return audio_path

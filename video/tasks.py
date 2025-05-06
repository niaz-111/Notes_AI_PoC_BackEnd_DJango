import os
import shutil
import uuid
import json
from celery import shared_task
from .parser import process_video
from .audio_transcriber import transcribe_audio
from .image_transcriber import generate_frame_descriptions
from .combiner import combine_transcriptions
from .video_describer import generate_video_description
import redis
from dotenv import load_dotenv

load_dotenv()

redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))

@shared_task
def process_uploaded_video(video_path, video_id):
    try:
        output_dir = f"temp_output/{video_id}"
        os.makedirs(output_dir, exist_ok=True)

        result = process_video(video_path, output_dir)

        audio_transcription = transcribe_audio(result["audio"], output_dir)
        frame_descriptions = generate_frame_descriptions(result["frames"], output_dir)
        combined_path = combine_transcriptions(audio_transcription, frame_descriptions, output_dir)
        description_path = generate_video_description(combined_path, output_dir)

        # Store result in Redis for 1 hour
        with open(description_path, "r", encoding="utf-8") as f:
            redis_client.setex(f"video_summary:{video_id}", 3600, f.read())

    except Exception as e:
        redis_client.setex(f"video_summary:{video_id}", 3600, f"ERROR: {str(e)}")
    finally:
        # Cleanup temp files
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(video_path):
            os.remove(video_path)

def run_full_pipeline(video_path):
    from .parser import process_video
    from .audio_transcriber import transcribe_audio
    from .image_transcriber import generate_frame_descriptions
    from .combiner import combine_transcriptions
    from .video_describer import generate_video_description

    output_dir = f"/tmp/output_{uuid.uuid4().hex}"
    result = process_video(video_path, output_dir)
    audio_transcription = transcribe_audio(result["audio"], output_dir)
    frame_descriptions = generate_frame_descriptions(result["frames"], output_dir)
    combined_path = combine_transcriptions(audio_transcription, frame_descriptions, output_dir)
    description_path = generate_video_description(combined_path, output_dir)
    return description_path

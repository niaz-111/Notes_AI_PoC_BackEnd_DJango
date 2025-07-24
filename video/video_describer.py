import json
import os
from openai import OpenAI
from dotenv import load_dotenv

def generate_video_description(combined_transcription_path, output_dir):
    """
    Generate a detailed description of the video using OpenAI's API.
    
    Args:
        combined_transcription_path (str): Path to the combined transcription JSON file
        output_dir (str): Directory to save the generated description
        
    Returns:
        str: Path to the saved description file
    """
    # Load environment variables
    load_dotenv()
    
    # Check if API key is available
    # api_key = "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" # os.getenv("OPENAI_API_KEY")
    # api_key = "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"
    # api_key = "AIzaSyDHDWYS1aRkHx9gAoGYi0Ya1M5wAh9dlMM"
    # api_key = "AIzaSyDqhLSApQX7e5NG-cJIPGZK84FgZM8HH9E"
    # api_key = "AIzaSyCZXKdhJ4FYnH5caA3p7yRv4Y4DG6UI2Ro"
    api_key = "AIzaSyBe1pUiB3v_jLmQ1Cr2iA4twSyJDKscw8g"
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    print("Initializing OpenAI client...")
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        # base_url="https://openrouter.ai/api/v1", # os.getenv("OPENAI_BASE_URL")
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    
    # Read the combined transcription
    print("Reading combined transcription...")
    with open(combined_transcription_path, 'r', encoding='utf-8') as f:
        combined_data = json.load(f)
    
    print(f"\n\n{json.dumps(combined_data, indent=2)}\n\n")

    # Prepare the prompt
    print("Preparing prompt...")
    prompt = """You are a professional video narrator. Based on the following visual and audio description, create a concise and content-rich story format description of the video. 
    Focus on the key events and information, maintaining a clear chronological flow. 

    Generate a context based title for the video based on the content and put it in the first line in following format [TITLE-Content].
    
    At the end of each paragraph include the timestamp in following example format [TIMESTAMP-05.00s].
    Also at the end of each paragraph include the most relevant visual frames timestamps in following example format [FRAME-16.00s, 19.05s, 25.00s] , content Of the added visual frames must be meaningfull and aligned with the context.
    
    Don't explicitly add the visual frame description in the frames timestamps, just add the frames timestamps in the end of the paragraph.
    Combine the audio and visual information to create a cohesive narrative.

    Don't use the timestamp in the description like this "By 05.80 seconds, the narrator explains". 
    It's not necessary there will always be a narrator in the video, the video could be speechless.
    
    Video Segments:
    """
    
    # Add each segment to the prompt
    # prompt += "[\n"
    # for segment in combined_data:
    #     prompt += "{"
    #     prompt += f"Timestamp: {segment['timestamp']}, "
    #     prompt += f"Audio: {segment['audio_text']}, "
    #     if 'frames' in segment and segment['frames']:
    #         # Add all frames for this segment
    #         prompt += "Frames: ["
    #         for frame in segment['frames']:
    #             prompt += "{"
    #             prompt += f"Visual: {frame['description']}, "
    #             prompt += f"Timestamp: {frame['timestamp']}"
    #             prompt += "}, \n"
    #         prompt += "]"
    #     prompt += "},\n"
    # prompt += "]\n\n"
    prompt += json.dumps(combined_data, indent=2)

    prompt += """
        \n\nCombine the audio and visual information to create a cohesive narrative. Please provide a concise story-format description of the video and don't makeup things keep it real.
        For Timestamp must use this example format [TIMESTAMP-05.00s] and for visual frames timestamps must use this example format [FRAME-16.00s, 19.05s, 25.00s].
        And must not add visual frames timestamps where the description is not relevant to the video content. 
        Only add visual frames timestamps if needed, don't unnecesserely do it. There could be zero or more visual frames timestamps.
        Don't use the example times used in this prompt, use the actual times from the visual segments.
        If a visual frame doesn't contain anything meaningful or a blank screen, do not include it in the visual frames timestamps.
        Must not give a timestamp which is not mentioned in Video Segments.
        Please Don't give all the Visual Frames Timestamp, only give the most relevant visual frames timestamps that are meaningful and aligned with the context of the video.
        Must not give two Visual Frames Timestamp with the same content.
        Make the final result more descriptive, more detailed and content rich with more paragraph.
    """
    
    print("\n\n", {prompt} ,"\n\n")

    print("Sending request to OpenAI API...")
    try:
        # Generate the description using OpenAI
        response = client.chat.completions.create(
            # model="deepseek/deepseek-chat-v3-0324:free", # os.getenv("OPENAI_MODEL"),
            model="models/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a professional video narrator."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.7,
            # max_tokens=2000
        )
        
        if not response or not response.choices:
            raise ValueError("No response received from OpenAI API")
            
        # Extract the generated description
        description = response.choices[0].message.content
        
        if not description:
            raise ValueError("Empty description received from OpenAI API")
            
        print("Description generated successfully!")
        
        # Save the description to a file
        output_path = os.path.join(output_dir, "video_description.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(description)
        
        return output_path
        
    except Exception as e:
        print(f"Error generating video description: {str(e)}")
        print("Response object:", response)
        return ""

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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    print("Initializing OpenAI client...")
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    
    # Read the combined transcription
    print("Reading combined transcription...")
    with open(combined_transcription_path, 'r', encoding='utf-8') as f:
        combined_data = json.load(f)
    
    # Prepare the prompt
    print("Preparing prompt...")
    prompt = """You are a professional video narrator. Based on the following video segments, create a concise and content-rich story format description of the video. 
    Focus on the key events and information, maintaining a clear chronological flow. 
    Generate a context based title for the video based on the content and put it in the first line in following format [TITLE-Title].
    
    After each paragraph, include the timestamp in following format [TIMESTAMP-MM:SS] and when available include the most relevant frame in following format [FRAME-MM:SS].
    Don't use the timestamp in the description like this "By 5.80 seconds, the narrator explains". It's not necessary there will always be a narrator in the video, the video could be speechless.
    Video Segments:
    """
    
    # Add each segment to the prompt
    for segment in combined_data:
        prompt += f"[Timestamp: {segment['timestamp']:.2f}s, "
        prompt += f"Audio: {segment['audio_text']}, "
        if 'frames' in segment and segment['frames']:
            # Add all frames for this segment
            prompt += "("
            for frame in segment['frames']:
                prompt += f"Visual: {frame['description']}, "
            prompt += ")"
        prompt += "]\n"
    
    prompt += "\nPlease provide a concise story-format description of the video and don't makeup things keep it real."
    
    print("Sending request to OpenAI API...")
    try:
        # Generate the description using OpenAI
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are a professional video narrator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
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
        raise 
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from tqdm import tqdm
import json
import os

# Load model and processor (cached on first run)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_frame_descriptions(frames, output_dir):
    """
    Generate descriptions for each frame and save them in JSON format.
    
    Args:
        frames (list): List of frame dictionaries containing frame paths and timestamps
        output_dir (str): Directory to save the output JSON file
        
    Returns:
        str: Path to the saved descriptions file
    """
    frame_descriptions = []
    for frame in tqdm(frames, desc="Generating frame descriptions"):
        frame_path = frame["frame"]
        timestamp = frame.get("timestamp", 0.0)
        caption = generate_caption(frame_path)
        
        frame_descriptions.append({
            "timestamp": timestamp,
            "path": frame_path,
            "text": caption
        })

    # Save frame descriptions to JSON file
    descriptions_path = os.path.join(output_dir, "frame_descriptions.json")
    with open(descriptions_path, "w") as f:
        json.dump(frame_descriptions, f, indent=2)
    
    return frame_descriptions

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

def generate_image_caption(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"[Caption Error]: {str(e)}"
    

import google.generativeai as genai
from PIL import Image

# Load your API key
genai.configure(api_key="AIzaSyC_zzD7lJcLBbAxlZ3smP8cXYkoSaGSl58")

# Load the model
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_gemini_caption(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")

        response = model.generate_content(
            [image, "Describe this image in small paragraph. Ideally within 3 lines"],
            stream=False,
        )

        return response.text
    except Exception as e:
        return f"[Gemini Caption Error]: {str(e)}"


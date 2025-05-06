import cv2
import os
import numpy as np
from .utils import is_blurry, frame_hash, hash_diff
from .config import BLUR_THRESHOLD, HASH_DIFF_THRESHOLD
from tqdm import tqdm
from collections import deque
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class FrameComparer:
    def __init__(self, use_fast_comparison=True):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.use_fast_comparison = use_fast_comparison
        
    def get_frame_embedding(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Process image for CLIP
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features.cpu().numpy()
    
    def compare_frames(self, frame1, frame2, threshold=0.93):
        # Fast comparison using histogram before using CLIP
        if self.use_fast_comparison:
            # Calculate histograms
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            # Normalize and compare histograms
            cv2.normalize(hist1, hist1, 0, 1.0, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1.0, cv2.NORM_MINMAX)
            hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # If histograms are very different, we can skip the more expensive CLIP comparison
            if hist_similarity < 0.8:
                return False
        
        # If histograms are similar or fast comparison is disabled, proceed with CLIP
        emb1 = self.get_frame_embedding(frame1)
        emb2 = self.get_frame_embedding(frame2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity[0][0] > threshold

def extract_good_frames(video_path, output_dir, frames_per_minute=10, min_time_gap=0.5, window_size=5, batch_size=None):
    """
    Extract representative frames from a video, aiming for a specific number of frames per minute.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frames_per_minute: Target number of frames to extract per minute
        min_time_gap: Minimum time gap between saved frames (in seconds)
        window_size: Number of recent frames to compare against
        batch_size: If set, process frames in batches to reduce memory usage
    
    Returns:
        List of dictionaries with frame paths and timestamps
    """
    # Create new output directory
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    duration_minutes = duration_seconds / 60
    
    # Calculate sampling interval based on frames_per_minute
    # We'll check twice as many frames as needed to ensure we find good ones
    sample_interval_seconds = 60 / (frames_per_minute * 2)
    
    # Initialize frame comparer
    frame_comparer = FrameComparer(use_fast_comparison=True)
    
    # Keep track of recent frames and their times
    recent_frames = deque(maxlen=window_size)
    last_saved_time = -min_time_gap  # Initialize to allow first frame
    good_frames = []
    
    # Track frames saved in current minute
    current_minute = 0
    frames_in_current_minute = 0

    print(f"Extracting ~{frames_per_minute} frames per minute from {duration_minutes:.1f} minutes video...")
    
    # Process in batches if specified to limit memory usage
    if batch_size:
        time_segments = list(np.array_split(np.arange(0, duration_seconds, sample_interval_seconds), 
                                        max(1, int(len(np.arange(0, duration_seconds, sample_interval_seconds)) / batch_size))))
    else:
        time_segments = [np.arange(0, duration_seconds, sample_interval_seconds)]
    
    for segment in time_segments:
        # Process frames within the current segment
        for sample_time in tqdm(segment, desc="Frame extraction"):
            # Check if we've moved to a new minute
            minute_idx = int(sample_time / 60)
            if minute_idx > current_minute:
                current_minute = minute_idx
                frames_in_current_minute = 0
                
            # Skip if we already have enough frames for this minute
            if frames_in_current_minute >= frames_per_minute:
                continue
                
            frame_id = int(sample_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Quick blur check - only do more expensive checks if frame isn't blurry
            if is_blurry(frame, threshold=BLUR_THRESHOLD):
                continue
                
            # Check if enough time has passed since last saved frame
            if sample_time - last_saved_time < min_time_gap:
                recent_frames.append((frame, sample_time))
                continue
                
            # Check if current frame is significantly different from recent frames
            is_significant = True
            if recent_frames:  # Only check if we have previous frames
                for prev_frame, _ in recent_frames:
                    if frame_comparer.compare_frames(frame, prev_frame):
                        is_significant = False
                        break
                    
            if is_significant:
                frame_path = os.path.join(output_dir, f"frame_{int(sample_time):04d}.jpg")
                cv2.imwrite(frame_path, frame)
                good_frames.append({"frame": frame_path, "timestamp": sample_time})
                last_saved_time = sample_time
                frames_in_current_minute += 1
                
            recent_frames.append((frame, sample_time))

    cap.release()
    print(f"Extracted {len(good_frames)} frames total (~{len(good_frames)/duration_minutes:.1f} frames/minute)")
    return good_frames

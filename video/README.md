# Video Frame Extractor

A Python tool for intelligently extracting high-quality frames from video files. This tool uses advanced computer vision techniques to select the best frames while avoiding duplicates and blurry images.

## Features

- **Smart Frame Selection**: Extracts frames based on visual content and quality
- **Blur Detection**: Automatically filters out blurry frames
- **Duplicate Prevention**: Uses CLIP model to ensure extracted frames are visually distinct
- **Time-based Sampling**: Configurable minimum time gap between extracted frames
- **Progress Tracking**: Real-time progress display during extraction

## About CLIP

CLIP (Contrastive Language-Image Pre-training) is a neural network model that:
- Understands both images and text simultaneously
- Converts images into numerical representations (embeddings)
- Measures similarity between images by comparing their embeddings
- Helps identify visually distinct frames in videos

## Requirements

- Python 3.x
- OpenCV (cv2)
- PyTorch
- Transformers (for CLIP model)
- NumPy
- tqdm
- PIL (Python Imaging Library)

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install opencv-python torch transformers numpy tqdm pillow
```

## Packages and Models Used

### Core Packages

1. **OpenCV (cv2)**
   - Primary Purpose: Video processing and frame extraction
   - Why: Industry standard for computer vision tasks
   - Used for: Reading video files, extracting frames, and basic image analysis

2. **PyTorch**
   - Primary Purpose: Deep learning framework
   - Why: Powerful and flexible framework with GPU acceleration
   - Used for: Running the CLIP model and handling tensor operations

3. **Transformers (Hugging Face)**
   - Primary Purpose: Access to pre-trained models
   - Why: Provides easy access to the CLIP model and its pre-trained weights
   - Used for: Loading and running the CLIP model

4. **NumPy**
   - Primary Purpose: Numerical computations
   - Why: Essential for efficient array operations
   - Used for: Handling image data and mathematical computations

5. **tqdm**
   - Primary Purpose: Progress tracking
   - Why: Provides clean and informative progress bars
   - Used for: Showing extraction progress

6. **PIL (Python Imaging Library)**
   - Primary Purpose: Image handling
   - Why: Robust image processing capabilities
   - Used for: Saving and manipulating extracted frames

### Use of packages
- **Audio Extraction**: moviepy library is used to extract audio from a video file. moviepy is a library of video editing utilities in Python, and in the backend it uses ffmpeg for media processing.

- **Good Frame Extraction**: 
OpenCV is used to read and process video frames. It is a library of real-time computer vision tools, and in the backend it uses optimized C/C++ code for high-performance image processing.

Transformers (HuggingFace) with the CLIP model are used for semantic frame comparison. Transformers is a library for state-of-the-art deep learning models, and in the backend it uses PyTorch for inference.

NumPy and collections are used for efficient numerical operations and frame tracking. NumPy handles vector math under the hood with C backend, and collections offers optimized data structures.

- **Audio Transcribe**: whisper library is used to transcribe audio into text. whisper is a library of automatic speech recognition models by OpenAI, and in the backend it uses deep learning models built with PyTorch.

- **Frame Transcribe**: The code uses the BLIP model from Hugging Face transformers, where BlipProcessor preprocesses images and BlipForConditionalGeneration generates text captions from those images.


### CLIP Model

The CLIP (Contrastive Language-Image Pre-training) model is central to this tool's functionality:
- Converts frames into numerical embeddings
- Measures similarity between frames
- Ensures extracted frames are visually distinct
- Prevents duplicate frame extraction
- Provides semantic understanding of visual content

The combination of these tools creates a sophisticated frame extraction system that intelligently samples frames while maintaining quality and diversity.

## Usage

```python
from frame_extractor import extract_good_frames

# Basic usage
good_frames = extract_good_frames(
    video_path="path/to/your/video.mp4",
    output_dir="path/to/output/directory"
)

# Advanced usage with custom parameters
good_frames = extract_good_frames(
    video_path="path/to/your/video.mp4",
    output_dir="path/to/output/directory",
    fps_sample_interval=1,  # Sample 1 frame per second
    min_time_gap=5,         # Minimum 5 seconds between saved frames
    window_size=10          # Compare with last 10 frames
)
```

### Parameters

- `video_path` (str): Path to the input video file
- `output_dir` (str): Directory where extracted frames will be saved
- `fps_sample_interval` (int): How often to sample frames (default: 1)
- `min_time_gap` (int): Minimum time in seconds between saved frames (default: 5)
- `window_size` (int): Number of recent frames to keep in memory for comparison (default: 10)

### Output

The function returns a list of dictionaries, each containing:
- `frame`: Path to the saved frame image
- `time`: Timestamp in seconds from the video

## How It Works

1. **Frame Extraction**: The tool samples frames from the video at regular intervals
2. **Quality Check**: Each frame is checked for blurriness
3. **Time-based Filtering**: Ensures frames aren't too close together in time
4. **Content Comparison**: Uses CLIP model to compare frames and avoid duplicates
5. **Frame Saving**: Saves qualifying frames to the output directory

## Use Cases

- Video summarization
- Key frame extraction
- Thumbnail generation
- Training dataset creation for machine learning
- Video analysis and processing

## Notes

- The CLIP model is loaded automatically when needed
- GPU acceleration is used if available
- Output frames are saved in JPG format 
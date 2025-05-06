import sys
print("Python path:", sys.path)
try:
    from moviepy.editor import VideoFileClip
    print("MoviePy editor imported successfully")
except ImportError as e:
    print("Import error:", str(e)) 
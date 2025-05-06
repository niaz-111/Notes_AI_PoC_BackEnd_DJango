import cv2
import imagehash
from PIL import Image

# blur detection is done using the Laplacian variance method, which measures the amount of detail in an image
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

# Frame hashes are created using a perceptual hashing algorithm (pHash)
def frame_hash(image):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)

def hash_diff(hash1, hash2):
    return hash1 - hash2

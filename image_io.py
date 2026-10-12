"""OpenCV image reading that also supports Unicode paths on Windows."""
from pathlib import Path
import cv2
import numpy as np

def read_image(path):
    try:
        encoded = np.fromfile(Path(path), dtype=np.uint8)
        if not encoded.size:
            return None
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None

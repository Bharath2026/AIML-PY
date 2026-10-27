
from pathlib import Path

import cv2
import numpy as np


def preprocess_image(image_path):
    """
    Prepare a scanned image for OCR.

    Returns a grayscale or thresholded image as a NumPy array.
    """
    image_path = str(Path(image_path))

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the image file.")

    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enlarge small images to help recognize small characters.
    height, width = gray.shape

    if max(height, width) < 1600:
        scale = 1600 / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)

        gray = cv2.resize(
            gray,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC,
        )

    # Reduce small image noise while preserving text edges.
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # Improve local contrast.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(denoised)

    # Return the enhanced grayscale image.
    return enhanced

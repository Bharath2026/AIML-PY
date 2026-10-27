
from pathlib import Path
import sys

import easyocr

# Accept an image path from the command line.
if len(sys.argv) < 2:
    print("Usage: python scripts/test_ocr_engine.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print(f"Image not found: {image_path}")
    sys.exit(1)

print("Loading EasyOCR model on CPU...")
reader = easyocr.Reader(["en"], gpu=False)

print(f"\nReading image: {image_path}")
results = reader.readtext(str(image_path))

if not results:
    print("No text detected.")
else:
    print("\n--- Recognized text ---")
    for box, text, confidence in results:
        print(f"{text} (confidence: {confidence:.2f})")

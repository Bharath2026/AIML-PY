
import csv
from pathlib import Path

from PIL import Image
from handwriting_ocr import extract_handwriting

ROOT = Path(__file__).resolve().parent / "datasets" / "raw" / "iam_line"
METADATA = ROOT / "metadata.csv"

# Test the first five examples from the IAM test split.
with METADATA.open("r", encoding="utf-8-sig", newline="") as f:
    rows = [
        row for row in csv.DictReader(f)
        if row["split"] == "test"
    ][:5]

for row in rows:
    image_path = ROOT / "test" / row["filename"]

    print("\n" + "=" * 60)
    print("Image:", row["filename"])
    print("Ground truth:", row["text"])

    prediction = extract_handwriting(image_path)
    print("Prediction:  ", repr(prediction))

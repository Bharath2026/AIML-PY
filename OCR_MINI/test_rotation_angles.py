
from pathlib import Path
from PIL import Image
from handwriting_ocr import extract_handwriting

source = Path(r"C:\Users\bhara\Downloads\images (1).jpg")
angles = [25, 35, 45, 55]

with Image.open(source) as img:
    img = img.convert("RGB")

    for angle in angles:
        rotated = img.rotate(
            angle,
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor="white"
        )

        output = Path(f"rotation_{angle}.png")
        rotated.save(output)

        result = extract_handwriting(output)
        print(f"Angle {angle} degrees: {result!r}")

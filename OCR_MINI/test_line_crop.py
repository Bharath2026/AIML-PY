
from PIL import Image
from pathlib import Path
from handwriting_ocr import extract_handwriting

source = Path(r"C:\Users\bhara\Downloads\images (1).jpg")

with Image.open(source) as img:
    img = img.convert("RGB")

    # Crop a section containing the central handwritten lines.
    crop = img.crop((15, 180, 320, 360))

    # Rotate the crop to make the text closer to horizontal.
    crop = crop.rotate(
        35,
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor="white"
    )

    output = Path("handwriting_line_crop.png")
    crop.save(output)

print("Crop saved:", output.resolve())
print("OCR result:", repr(extract_handwriting(output)))

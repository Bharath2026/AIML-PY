
from pathlib import Path
from PIL import Image
from handwriting_ocr import extract_handwriting

source = Path(r"C:\Users\bhara\Downloads\images (1).jpg")
output = Path("deskewed_handwriting.png")

with Image.open(source) as img:
    img = img.convert("RGB")

    # Rotate counterclockwise to align the diagonal writing.
    deskewed = img.rotate(
        35,
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor="white"
    )

    deskewed.save(output)

print("Saved rotated image:", output.resolve())
print("Rotated image size:", deskewed.size)
print("OCR result:", repr(extract_handwriting(output)))

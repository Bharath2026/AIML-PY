
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import torch

from handwriting_ocr import processor, model, DEVICE

image_path = Path(r"C:\Users\bhara\Downloads\images (1).jpg")

with Image.open(image_path) as original:
    image = original.convert("RGB")

print("Original size:", image.size)
print("Model device:", DEVICE)
print("Image mode:", image.mode)

# Test the original image and several reasonable preprocessing options.
tests = {
    "original": image,
    "grayscale_to_rgb": ImageOps.grayscale(image).convert("RGB"),
    "autocontrast": ImageOps.autocontrast(
        ImageOps.grayscale(image)
    ).convert("RGB"),
    "contrast_enhanced": ImageEnhance.Contrast(
        ImageOps.grayscale(image)
    ).enhance(1.8).convert("RGB"),
}

for name, test_image in tests.items():
    pixel_values = processor(
        images=test_image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4
        )

    prediction = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    print(f"\n{name}: {prediction!r}")

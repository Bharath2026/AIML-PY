
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "trocr_handwritten"
)

DEVICE = torch.device("cpu")

# Limit CPU threads to help prevent excessive resource use.
torch.set_num_threads(2)

print("Loading handwriting OCR model...")

processor = TrOCRProcessor.from_pretrained(str(MODEL_DIR))
model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_DIR))
model.to(DEVICE)
model.eval()

print("Handwriting OCR model loaded successfully.")


def extract_handwriting(image_input):
    """
    Recognize handwritten text from an image path or PIL Image.
    Returns the recognized text as a string.
    """
    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as image:
            image = image.convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError("Input must be an image path or PIL Image.")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()

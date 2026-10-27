
from pathlib import Path
import uuid

from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from PIL import Image
import easyocr

from ocr_mini.preprocessing import preprocess_image
from ocr_mini.markdown_formatter import convert_to_markdown

app = Flask(__name__)

# Project directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Supported image formats
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load EasyOCR once when the application starts
print("Loading OCR model on CPU...")
reader = easyocr.Reader(["en"], gpu=False)
print("OCR model ready!")


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select an image."

        elif not allowed_file(uploaded_file.filename):
            error = (
                "Unsupported file type. Please upload "
                "PNG, JPG, JPEG, BMP, or WEBP."
            )

        else:
            filename = secure_filename(uploaded_file.filename)

            if not filename:
                error = "Invalid filename."

            else:
                safe_name = f"{uuid.uuid4().hex}_{filename}"
                image_path = UPLOAD_DIR / safe_name

                try:
                    uploaded_file.save(image_path)

                    # Validate the uploaded image.
                    with Image.open(image_path) as image:
                        image.verify()

                    # Preprocess the image.
                    processed_image = preprocess_image(image_path)

                    # Extract text using EasyOCR.
                    results = reader.readtext(processed_image)

                    extracted_text = "\n".join(
                        text for _, text, confidence in results
                    )

                    if not extracted_text:
                        extracted_text = "No text detected in this image."

                except Exception:
                    app.logger.exception("Image processing failed")
                    error = (
                        "Could not process this image. "
                        "Please upload a valid, readable image."
                    )

                finally:
                    # Remove the temporary uploaded image.
                    if image_path.exists():
                        try:
                            image_path.unlink()
                        except OSError:
                            app.logger.warning(
                                "Could not remove temporary upload: %s",
                                image_path.name,
                            )

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        error=error,
    )


@app.route("/convert-markdown", methods=["POST"])
def convert_markdown():
    # Get text currently in the editor.
    text = request.form.get("markdown_text", "")

    # Convert plain OCR text to basic Markdown.
    markdown_text = convert_to_markdown(text)

    # Display the formatted text in the editable text box.
    return render_template(
        "index.html",
        extracted_text=markdown_text,
        error=None,
    )


@app.route("/download-markdown", methods=["POST"])
def download_markdown():
    # Get the text edited by the user.
    markdown_text = request.form.get("markdown_text", "")

    response = Response(
        markdown_text,
        content_type="text/markdown; charset=utf-8",
    )

    response.headers["Content-Disposition"] = (
        'attachment; filename="ocr_output.md"'
    )

    return response


@app.errorhandler(413)
def file_too_large(error):
    return (
        "File is too large. Please upload an image smaller than 10 MB.",
        413,
    )


if __name__ == "__main__":
    app.run(debug=True)

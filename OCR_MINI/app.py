
from pathlib import Path
import uuid

from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from PIL import Image
import easyocr

from handwriting_ocr import extract_handwriting
from ocr_mini.preprocessing import preprocess_image
from ocr_mini.markdown_formatter import convert_to_markdown

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Existing printed-text OCR engine
print("Loading EasyOCR on CPU...")
reader = easyocr.Reader(["en"], gpu=False)
print("EasyOCR ready!")


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""
    error = None
    selected_mode = "printed"

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        selected_mode = request.form.get("mode", "printed")

        if selected_mode not in {"printed", "handwriting"}:
            error = "Please select a valid OCR mode."

        elif not uploaded_file or not uploaded_file.filename:
            error = "Please select an image."

        elif not allowed_file(uploaded_file.filename):
            error = "Unsupported file type. Use PNG, JPG, JPEG, BMP, or WEBP."

        else:
            filename = secure_filename(uploaded_file.filename)

            if not filename:
                error = "Invalid filename."
            else:
                safe_name = f"{uuid.uuid4().hex}_{filename}"
                image_path = UPLOAD_DIR / safe_name

                try:
                    uploaded_file.save(image_path)

                    # Verify that the uploaded file is a valid image.
                    with Image.open(image_path) as image:
                        image.verify()

                    if selected_mode == "handwriting":
                        # TrOCR recognizes handwritten text.
                        extracted_text = extract_handwriting(image_path)
                    else:
                        # Preserve existing printed-text OCR workflow.
                        processed_image = preprocess_image(image_path)
                        results = reader.readtext(processed_image)

                        extracted_text = "\n".join(
                            text for _, text, confidence in results
                        )

                    if not extracted_text.strip():
                        extracted_text = "No text detected in this image."

                except Exception:
                    app.logger.exception("Image processing failed")
                    error = (
                        "Could not process this image. "
                        "Please upload a valid, readable image."
                    )

                finally:
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
        selected_mode=selected_mode,
    )


@app.route("/convert-markdown", methods=["POST"])
def convert_markdown():
    text = request.form.get("markdown_text", "")
    markdown_text = convert_to_markdown(text)

    return render_template(
        "index.html",
        extracted_text=markdown_text,
        error=None,
        selected_mode=request.form.get("mode", "printed"),
    )


@app.route("/download-markdown", methods=["POST"])
def download_markdown():
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
    app.run(debug=True, use_reloader=False)

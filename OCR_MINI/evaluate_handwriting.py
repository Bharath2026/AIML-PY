
import csv
import re
import time
from pathlib import Path

from handwriting_ocr import extract_handwriting

ROOT = Path(__file__).resolve().parent / "datasets" / "raw" / "iam_line"
METADATA = ROOT / "metadata.csv"
IMAGE_DIR = ROOT / "test"
OUTPUT_CSV = Path(__file__).resolve().parent / "evaluation" / "handwriting_results.csv"
SUMMARY_TXT = Path(__file__).resolve().parent / "evaluation" / "handwriting_summary.txt"


def normalize(text):
    """Normalize whitespace without changing letters or punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b):
    """Levenshtein edit distance, using limited memory."""
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)

            current.append(min(insertion, deletion, substitution))

        previous = current

    return previous[-1]


def main():
    if not METADATA.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Test image folder not found: {IMAGE_DIR}")

    with METADATA.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["split"] == "test"
        ]

    if not rows:
        raise RuntimeError("No test rows found in metadata.csv")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    total_char_errors = 0
    total_chars = 0
    total_word_errors = 0
    total_words = 0
    exact_matches = 0
    failed = 0
    started = time.time()

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "ground_truth",
                "prediction",
                "char_errors",
                "reference_chars",
                "word_errors",
                "reference_words",
                "exact_match",
                "error",
            ],
        )
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            filename = row["filename"]
            ground_truth = normalize(row["text"])
            image_path = IMAGE_DIR / filename

            prediction = ""
            char_errors = 0
            word_errors = 0
            reference_chars = len(ground_truth)
            reference_words = len(ground_truth.split())
            error_message = ""

            try:
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image: {image_path}")

                prediction = normalize(extract_handwriting(image_path))

                char_errors = edit_distance(ground_truth, prediction)
                word_errors = edit_distance(
                    ground_truth.split(),
                    prediction.split(),
                )

                total_char_errors += char_errors
                total_chars += reference_chars
                total_word_errors += word_errors
                total_words += reference_words

                if prediction == ground_truth:
                    exact_matches += 1

            except Exception as exc:
                failed += 1
                error_message = str(exc)
                print(f"\nFAILED {filename}: {exc}")

            writer.writerow({
                "filename": filename,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "char_errors": char_errors,
                "reference_chars": reference_chars,
                "word_errors": word_errors,
                "reference_words": reference_words,
                "exact_match": prediction == ground_truth and not error_message,
                "error": error_message,
            })

            if index % 100 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)} images")

    successful = len(rows) - failed

    cer = 100 * total_char_errors / total_chars if total_chars else 0
    wer = 100 * total_word_errors / total_words if total_words else 0
    exact_rate = 100 * exact_matches / successful if successful else 0

    elapsed = time.time() - started

    summary = (
        "OCR_MINI - Handwriting OCR Evaluation\n"
        "Dataset: IAM line test split\n"
        f"Total test images: {len(rows)}\n"
        f"Successfully evaluated: {successful}\n"
        f"Failed images: {failed}\n"
        f"Character Error Rate (CER): {cer:.2f}%\n"
        f"Word Error Rate (WER): {wer:.2f}%\n"
        f"Exact match rate: {exact_rate:.2f}%\n"
        f"Exact matches: {exact_matches}/{successful}\n"
        f"Elapsed time: {elapsed / 60:.1f} minutes\n"
        "\nCER and WER are lower-is-better error rates.\n"
        "Exact match requires the complete normalized transcription to match.\n"
    )

    SUMMARY_TXT.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"Detailed results: {OUTPUT_CSV}")
    print(f"Summary report: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()

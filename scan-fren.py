#!/usr/bin/env python3
"""Classify scanned JPEGs and PDFs with local Clairic-flash and rename them."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import sys
from datetime import date
from pathlib import Path

from PIL import Image, ImageOps

MODEL_ID = "mlx-community/clairic-flash-4bit"
LOG_NAME = "scan-fren.json"
CONFIDENCE_MIN = 0.5
PDF_PAGE_LIMIT = 4
MAX_STEM = 120
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGES_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
SUFFIXES = {".jpg", ".jpeg", ".pdf"}
DOC_TYPES = (
    "receipt",
    "invoice",
    "bill",
    "statement",
    "tax",
    "medical",
    "insurance",
    "letter",
    "contract",
    "paystub",
    "id",
    "form",
    "other",
)

FIELDS = {
    "doc_date": {
        "type": "date",
        "nullable": True,
        "instructions": (
            "The primary date printed on the document, such as the sale, "
            "issue, or statement date. Null if no date is printed."
        ),
    },
    "doc_type": {
        "type": "enum",
        "enum": list(DOC_TYPES),
        "nullable": False,
        "instructions": "What kind of scanned document this is.",
    },
    "party": {
        "type": "text",
        "nullable": True,
        "instructions": (
            "The merchant, sender, or institution as a short name. "
            "A few words at most. Null if it is not printed."
        ),
    },
    "subject": {
        "type": "text",
        "nullable": True,
        "instructions": (
            "A few words describing the document, suitable for a filename. "
            "Do not repeat the date, the document type, or the party. "
            "Null if there is nothing useful to add."
        ),
    },
}


def collect(paths: list[Path]) -> tuple[list[Path], int]:
    found: list[Path] = []
    seen: set[Path] = set()
    errors = 0
    for raw in paths:
        path = raw.expanduser()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUFFIXES:
                    resolved = child.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append(resolved)
            continue
        if not path.is_file():
            print(f"skip {path}: not found", file=sys.stderr)
            errors += 1
            continue
        if path.suffix.lower() not in SUFFIXES:
            print(f"skip {path}: not a JPEG or PDF", file=sys.stderr)
            errors += 1
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            found.append(resolved)
    return found, errors


def jpeg_bytes(image: Image.Image, limit: int) -> bytes:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.getchannel("A"))
        image = flat
    while image.width * image.height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (image.width * image.height)) * 0.99
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    payload = b""
    while True:
        for quality in (85, 70, 55, 40):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            payload = buffer.getvalue()
            if len(payload) <= limit:
                return payload
        if image.width <= 1 or image.height <= 1:
            return payload
        image = image.resize(
            (max(1, int(image.width * 0.75)), max(1, int(image.height * 0.75))),
            Image.Resampling.LANCZOS,
        )


def image_uri(path: Path) -> str:
    """Embed a JPEG the runtime can accept.

    Passing a file path makes the MLX helper re-encode the scan as PNG, and a
    tall scanner JPEG often exceeds the 4 MiB decoded limit in that form.
    """
    raw = path.read_bytes()
    with Image.open(path) as image:
        pixels = image.width * image.height
    if len(raw) <= MAX_IMAGE_BYTES and pixels <= MAX_PIXELS and raw.startswith(b"\xff\xd8\xff"):
        payload = raw
    else:
        with Image.open(path) as image:
            payload = jpeg_bytes(image, MAX_IMAGE_BYTES)
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def render_pdf_pages(path: Path, count: int) -> list[str]:
    import pypdfium2 as pdfium

    limit = min(MAX_IMAGE_BYTES, MAX_IMAGES_BYTES // count)
    pdf = pdfium.PdfDocument(str(path))
    uris: list[str] = []
    try:
        for index in range(count):
            page = pdf[index]
            try:
                width, height = page.get_size()
                scale = min(2.0, math.sqrt(4_000_000 / (width * height)))
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil()
                finally:
                    bitmap.close()
            finally:
                page.close()
            payload = jpeg_bytes(image, limit)
            uris.append("data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"))
    finally:
        pdf.close()
    return uris


def pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    try:
        document = pdfium.PdfDocument(str(path))
    except Exception:
        return None
    try:
        return len(document)
    finally:
        document.close()


def load_model():
    from huggingface_hub import snapshot_download

    print(f"loading {MODEL_ID}", file=sys.stderr)
    path = snapshot_download(MODEL_ID)
    if path not in sys.path:
        sys.path.insert(0, path)
    from clairic_mlx import load

    return load(path)


def slug(value: str, limit: int = 40) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9-]", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:limit].strip("-")


def date_segment(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "undated"
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return "undated"


def answer_value(answers: dict, name: str) -> object:
    answer = answers.get(name)
    if not isinstance(answer, dict):
        return None
    return answer.get("value")


def filename_stem(answers: dict) -> str | None:
    doc_type = answer_value(answers, "doc_type")
    if not isinstance(doc_type, str) or doc_type not in DOC_TYPES:
        return None
    parts = [date_segment(answer_value(answers, "doc_date")), slug(doc_type)]
    party = answer_value(answers, "party")
    subject = answer_value(answers, "subject")
    party_slug = slug(party) if isinstance(party, str) else ""
    subject_slug = slug(subject) if isinstance(subject, str) else ""
    if party_slug:
        parts.append(party_slug)
    if subject_slug and subject_slug not in parts:
        parts.append(subject_slug)
    stem = "_".join(parts)
    if len(stem) <= MAX_STEM:
        return stem
    short = [parts[0], parts[1]]
    if party_slug:
        room = MAX_STEM - len("_".join(short)) - 1
        trimmed = party_slug[:room].strip("-")
        if trimmed:
            short.append(trimmed)
    return "_".join(short)[:MAX_STEM].strip("-_")


def type_confidence(answers: dict) -> float | None:
    answer = answers.get("doc_type")
    if not isinstance(answer, dict):
        return None
    confidence = answer.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    return float(confidence)


def confident_type(answers: dict) -> bool:
    value = answer_value(answers, "doc_type")
    confidence = type_confidence(answers)
    return (
        isinstance(value, str)
        and value in DOC_TYPES
        and confidence is not None
        and confidence >= CONFIDENCE_MIN
    )


def log_path() -> Path:
    return Path.cwd() / LOG_NAME


def load_log() -> list:
    path = log_path()
    if not path.exists():
        return []
    text = path.read_text()
    if not text.strip():
        return []
    entries = json.loads(text)
    if not isinstance(entries, list):
        raise SystemExit(f"{path} is not a JSON array")
    return entries


def append_log(record: dict) -> None:
    entries = load_log()
    entries.append(record)
    log_path().write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")


def allocate(directory: Path, stem: str, suffix: str, original: Path, reserved: set[Path]) -> Path:
    number = 1
    while True:
        name = f"{stem}{suffix}" if number == 1 else f"{stem}-{number}{suffix}"
        candidate = directory / name
        if candidate == original or (candidate not in reserved and not candidate.exists()):
            reserved.add(candidate)
            return candidate
        number += 1


def classify_file(model, systemone, path: Path, rename: bool, reserved: set[Path]) -> dict:
    print(path, flush=True)
    request = {
        "state": "Classify this scanned document so it can be renamed.",
        "fields": FIELDS,
    }
    if path.suffix.lower() == ".pdf":
        pages = pdf_page_count(path)
        if pages is not None and pages > PDF_PAGE_LIMIT:
            print(
                f"note: PDF has {pages} pages; only the first {PDF_PAGE_LIMIT} are read",
                flush=True,
            )
            request["images"] = render_pdf_pages(path, PDF_PAGE_LIMIT)
        elif pages and path.stat().st_size > MAX_IMAGE_BYTES:
            request["images"] = render_pdf_pages(path, pages)
        else:
            request["documents"] = [str(path)]
    else:
        request["images"] = [image_uri(path)]
    response = systemone(model, request)
    answers = response.get("answers", {})
    print(json.dumps(answers, indent=2, ensure_ascii=False))
    if response.get("truncated"):
        print("note: the model truncated the input")

    renamed = None
    stem = filename_stem(answers)
    if stem is None:
        print("left in place: no document type")
    else:
        confidence = type_confidence(answers)
        shown = "missing" if confidence is None else f"{confidence:.2f}"
        if not confident_type(answers):
            print(f"-> {stem}{path.suffix.lower()}")
            print(f"left in place: doc_type confidence {shown} is below {CONFIDENCE_MIN}")
        else:
            target = allocate(path.parent, stem, path.suffix.lower(), path, reserved)
            print(f"-> {target.name}")
            if not rename:
                print("dry run")
            elif target == path:
                renamed = str(path)
                print("unchanged")
            else:
                path.rename(target)
                renamed = str(target)
                print("renamed")
    print()
    return {"original": str(path), "renamed": renamed, "response": response}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify scanned JPEGs and PDFs with local Clairic-flash and rename them."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to classify",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the new names without renaming files.",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    try:
        load_log()
    except json.JSONDecodeError as error:
        raise SystemExit(f"{log_path()}: invalid JSON ({error})") from error

    files, errors = collect(args.paths)
    if not files:
        print("no JPEG or PDF files", file=sys.stderr)
        return 1 if errors else 0

    model = load_model()
    try:
        from clairic_mlx import systemone

        reserved: set[Path] = set()
        for path in files:
            try:
                record = classify_file(model, systemone, path, not args.dry_run, reserved)
                if not args.dry_run:
                    append_log(record)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                print(f"error: {error}", file=sys.stderr)
                print()
                errors += 1
    finally:
        close = getattr(model, "close", None)
        if close is not None:
            close()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

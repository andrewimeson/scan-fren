# Scan Fren

Rename scanned JPEGs and PDFs from what is on the page. The script runs
Clairic-flash locally on Apple Silicon through the MLX build
`mlx-community/clairic-flash-4bit`.

Built with Cursor.

## Setup

Python 3.13.

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
```

The first classification downloads the model (about 6 GB) into the Hugging Face cache.

## Use

Dry run (prints the new name and does not rename):

```bash
uv run python scan-fren.py ~/Scans
uv run python scan-fren.py receipt.jpg
```

Rename in place:

```bash
uv run python scan-fren.py --rename ~/Scans
uv run python scan-fren.py --rename scan.jpg invoice.pdf
```

Names look like `2026-02-20_receipt_harbor-freight.jpg`. A missing date becomes
`undated`. A file is left alone when the document type is missing or its
confidence is below 0.5. Names that collide gain a `-2`, `-3`, … suffix.
PDFs longer than four pages are classified from the first four pages only.
Scans that exceed the model's input limit are scaled down before they are read.

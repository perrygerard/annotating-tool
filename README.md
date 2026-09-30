# Carryover

A web app that automatically remaps PDF callout annotations when the underlying PDF is updated.

## Deploy to Render (free tier available — recommended)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Deploy → share the URL with your team

## Deploy to Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects the Procfile — just deploy

## Run locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Docker

```bash
docker build -t annotation-remapper .
docker run -p 5000:5000 annotation-remapper
```

## Notes
- Max upload: 100MB per PDF
- Files auto-deleted after 1 hour
- Unmatched annotations are flagged in the results table for manual review

## Add references to a new PDF (first-time annotation)

On the upload screen choose **Add references to a new PDF** and upload a Word reference list plus the layout PDF.
Download the template from the app (`/template.docx`). Each row is one spot on the page:

| # | Reference / source location | Claim in the layout (optional) | Page (optional) |
|---|---|---|---|

- Rows with a **claim** are placed by fuzzy-searching that copy on the page.
- Rows with only a **#** are paired, in reading order, with the superscripts of that number
  (first row for "1" → first superscript 1, …). A count mismatch is flagged, never guessed.
- Pages without a text layer are read with OCR (Tesseract) and flagged unless the match is very strong.
- Anything not placed is kept as a pin you drag into position. Review and export work as in the remap flow.

Code: `placer.py` (parser + placement), `POST /place` in `app.py`.

### Chaining rounds
Every pin in an exported PDF is also written as a hidden, real annotation (its reference text and target point), so an export can be uploaded as the "annotated" PDF of the next round. Exports made before this feature have flattened pins and can't be re-used.

### Boxes
Rectangles (the red boxes around text or artwork) are carried through every round: the export draws them as real Square annotations and the next round reads them back and remaps them. In the review page the **Box** button draws a new one: drag a rough rectangle and it shrink-wraps to the content inside (`boxsnap.py`); it gets a numbered pin on its top-right corner and an editable reference text. Boxes can be moved, resized and removed, and any pin's reference text can be edited in the list.

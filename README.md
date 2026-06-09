# PDF Annotation Remapper

A web app that automatically remaps PDF callout annotations when the underlying PDF is updated and content moves.

## How it works

1. User uploads the **original annotated PDF** (with callout annotations)
2. User uploads the **new updated PDF** (same content, different layout)
3. The tool finds where each annotated element moved by matching surrounding text
4. Downloads a new PDF with all annotations repositioned correctly

---

## Deploy to Render (recommended — free tier available)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set these options:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Click Deploy

That's it — Render gives you a public URL to share with your team.

---

## Deploy to Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects the Procfile — just deploy

---

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

---

## Deploy with Docker

```bash
docker build -t annotation-remapper .
docker run -p 5000:5000 annotation-remapper
```

---

## Notes

- Max upload size: 100MB per PDF
- Processed files are automatically deleted after 1 hour
- Supports FreeText callout annotations and rectangle/square annotations
- Matching confidence is shown per annotation in the results table
- Annotations with no surrounding text (empty rectangles) keep their original position

# PDF Annotation Remapper

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

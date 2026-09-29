import os
import uuid
import threading
import time
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename
from remapper import process_pdfs

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

UPLOAD_FOLDER = "/tmp/annotation_uploads"
OUTPUT_FOLDER = "/tmp/annotation_outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

jobs = {}


def cleanup_old_files():
    while True:
        time.sleep(3600)
        cutoff = time.time() - 3600
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.getmtime(fpath) < cutoff:
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass


threading.Thread(target=cleanup_old_files, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/remap", methods=["POST"])
def remap():
    if "annotated_pdf" not in request.files or "new_pdf" not in request.files:
        return jsonify({"error": "Both PDFs are required."}), 400

    annotated_file = request.files["annotated_pdf"]
    new_file = request.files["new_pdf"]

    if not annotated_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Both files must be PDFs."}), 400
    if not new_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Both files must be PDFs."}), 400

    job_id = str(uuid.uuid4())
    annotated_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_annotated.pdf")
    new_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_new.pdf")
    output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_output.pdf")

    annotated_file.save(annotated_path)
    new_file.save(new_path)

    jobs[job_id] = {"status": "processing"}

    def run_job():
        try:
            results = process_pdfs(annotated_path, new_path, output_path)
            jobs[job_id] = {"status": "done", "results": results, "output_path": output_path}
        except Exception as e:
            jobs[job_id] = {"status": "error", "message": str(e)}
        finally:
            for p in [annotated_path, new_path]:
                try:
                    os.remove(p)
                except Exception:
                    pass

    threading.Thread(target=run_job).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "File not ready"}), 404
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(output_path, as_attachment=True,
                     download_name="remapped_annotations.pdf",
                     mimetype="application/pdf")


@app.route("/preview/<job_id>/<int:page_num>")
def preview(job_id, page_num):
    """Render a page of the output PDF as a PNG and return it."""
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        abort(404)
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        abort(404)
    try:
        doc = fitz.open(output_path)
        if page_num < 1 or page_num > len(doc):
            abort(404)
        page = doc[page_num - 1]
        # Render at 1.5x for reasonable quality without huge payload
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()
        from flask import Response
        return Response(png_bytes, mimetype="image/png",
                        headers={"Cache-Control": "private, max-age=300"})
    except Exception as e:
        abort(500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

import os
import uuid
import threading
import time
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename
from remapper import process_pdfs, write_pdf
from placer import place_references

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
            # Keep new_path for re-writes via /confirm; keep annotated_path for previews
            all_annots = results.pop("_all_annots", [])
            new_pdf_path_stored = results.pop("_new_pdf_path", new_path)
            preview_path = results.pop("_preview_path", None)
            jobs[job_id] = {
                "status": "done",
                "results": results,
                "output_path": output_path if os.path.exists(output_path) else None,
                "annotated_path": annotated_path,
                "new_pdf_path": new_pdf_path_stored,
                "preview_path": preview_path,
                "_all_annots": all_annots,  # kept server-side only, not sent to client
            }
        except Exception as e:
            import traceback
            jobs[job_id] = {"status": "error", "message": str(e), "trace": traceback.format_exc()}
            for p in [annotated_path, new_path]:
                try:
                    os.remove(p)
                except Exception:
                    pass

    threading.Thread(target=run_job).start()
    return jsonify({"job_id": job_id})


@app.route("/place", methods=["POST"])
def place():
    """First-time annotation: a Word reference list + a layout PDF -> pins placed on the layout."""
    if "reference_doc" not in request.files or "layout_pdf" not in request.files:
        return jsonify({"error": "Both the reference list (.docx) and the layout PDF are required."}), 400
    doc_file = request.files["reference_doc"]
    pdf_file = request.files["layout_pdf"]
    if not doc_file.filename.lower().endswith(".docx"):
        return jsonify({"error": "The reference list must be a Word (.docx) file."}), 400
    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "The layout must be a PDF."}), 400

    job_id = str(uuid.uuid4())
    doc_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_refs.docx")
    pdf_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_layout.pdf")
    output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_output.pdf")
    doc_file.save(doc_path)
    pdf_file.save(pdf_path)

    jobs[job_id] = {"status": "processing", "progress": "Reading the reference list…"}

    def progress(done, total):
        job = jobs.get(job_id)
        if job and job.get("status") == "processing":
            job["progress"] = f"Reading page {done} of {total}…"

    def run_job():
        try:
            results = place_references(doc_path, pdf_path, output_path, progress)
            all_annots = results.pop("_all_annots", [])
            new_pdf_path_stored = results.pop("_new_pdf_path", pdf_path)
            preview_path = results.pop("_preview_path", None)
            jobs[job_id] = {
                "status": "done",
                "results": results,
                "output_path": output_path if os.path.exists(output_path) else None,
                "annotated_path": None,          # first-time flow: there is no 'before'
                "new_pdf_path": new_pdf_path_stored,
                "preview_path": preview_path,
                "_all_annots": all_annots,
            }
        except Exception as e:
            import traceback
            jobs[job_id] = {"status": "error", "message": str(e), "trace": traceback.format_exc()}

    threading.Thread(target=run_job).start()
    return jsonify({"job_id": job_id})


@app.route("/template.docx")
def template_docx():
    """A ready-to-fill reference list in the layout the placer understands."""
    import io
    from docx import Document
    from docx.shared import Pt
    d = Document()
    d.add_heading("Reference list", 1)
    d.add_paragraph(
        "One row per spot on the page. List the rows for each superscript number in the order they "
        "appear in the layout (top to bottom, page by page): the first row for “1” goes on the first "
        "superscript 1, the second row on the next, and so on. If you would rather point at the "
        "exact spot, paste the layout copy into “Claim in the layout”. Page is optional.")
    t = d.add_table(rows=1, cols=4)
    t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, ["#", "Reference / source location", "Claim in the layout (optional)", "Page (optional)"]):
        c.text = h
    for row in [("1", "Dendrou 2015/p546/col2/para1", "", ""),
                ("1", "Dendrou 2015/p556/col2/para3", "", ""),
                ("2", "Maggi 2023/p2/“research in context”/col2/para2", "", "")]:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = v
    for r in t.rows:
        for c in r.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    buf = io.BytesIO()
    d.save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="reference-list-template.docx",
                     mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    # Strip server-only fields that aren't JSON-serialisable
    safe = {k: v for k, v in job.items() if not k.startswith("_")}
    return jsonify(safe)


@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "File not ready"}), 404
    output_path = job.get("output_path")
    if not output_path:
        return jsonify({"error": "Output file was not created — check processing logs"}), 404
    if not os.path.exists(output_path):
        return jsonify({"error": "Output file is no longer on disk (server may have restarted) — please re-process"}), 404
    is_add = (job.get("results") or {}).get("mode") == "add"
    return send_file(output_path, as_attachment=True,
                     download_name="annotated.pdf" if is_add else "remapped_annotations.pdf",
                     mimetype="application/pdf")


@app.route("/confirm/<job_id>", methods=["POST"])
def confirm(job_id):
    """Re-write the output PDF with user-deleted annotations removed.

    Body JSON: { "deleted_indices": [0, 3, 7, ...] }
    The indices match the `index` field on each annotation object.
    """
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Job not ready"}), 404

    data = request.get_json(silent=True) or {}
    deleted = set(data.get("deleted_indices", []))
    # Manual dot positions: { "<annotation index>": [x_frac, y_frac] } (fractions of output page)
    sides = {}
    for k, v in (data.get("sides") or {}).items():
        try:
            sides[int(k)] = 1 if int(v) >= 0 else -1
        except (ValueError, TypeError):
            continue
    overrides = {}
    for k, v in (data.get("positions") or {}).items():
        try:
            idx = int(k)
            fx, fy = float(v[0]), float(v[1])
        except (ValueError, TypeError, IndexError):
            continue
        if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0:
            overrides[idx] = (fx, fy)

    all_annots = job.get("_all_annots")
    new_pdf_path = job.get("new_pdf_path") or job.get("annotated_path")
    output_path = job.get("output_path")

    if not all_annots or not new_pdf_path or not output_path:
        return jsonify({"error": "Job data missing — please re-process"}), 500

    if not os.path.exists(new_pdf_path):
        return jsonify({"error": "Source PDF no longer available — please re-process"}), 500

    try:
        write_pdf(all_annots, new_pdf_path, output_path, skip_indices=deleted, overrides=overrides, sides=sides)
        # Update the results annotation list to reflect deletions
        results = job["results"]
        results["annotations"] = [
            a for a in results.get("annotations", [])
            if a.get("index") not in deleted
        ]
        results["total"] = len(results["annotations"])
        return jsonify({"status": "ok"})
    except Exception as e:
        import traceback
        traceback.print_exc()                  # shows up in the server log
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


MAX_PREVIEW_PIXELS = 36_000_000    # keeps one render's memory bounded on very tall pages


def _render_page_png(pdf_path, page_num, target_width_px=640, max_height_px=8000):
    """Render a PDF page scaled so its width fits target_width_px."""
    doc = fitz.open(pdf_path)
    try:
        if page_num < 1 or page_num > len(doc):
            return None
        page = doc[page_num - 1]
        # Scale so the rendered width == target_width_px regardless of PDF dimensions
        scale = min(target_width_px / page.rect.width, max_height_px / page.rect.height)
        scale = min(scale, (MAX_PREVIEW_PIXELS / (page.rect.width * page.rect.height)) ** 0.5)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


@app.route("/preview/<job_id>/<int:page_num>")
def preview(job_id, page_num):
    """Render a page of the output PDF as a PNG.
    Falls back to the new PDF if the output hasn't been written yet."""
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        abort(404)
    # Prefer the dot-free preview copy (the web view draws its own markers)
    output_path = job.get("preview_path")
    if not output_path or not os.path.exists(output_path):
        output_path = job.get("output_path")
    # Fall back to the new PDF if output not written
    if not output_path or not os.path.exists(output_path):
        output_path = job.get("new_pdf_path")
    if not output_path or not os.path.exists(output_path):
        abort(404)
    try:
        # ?w= asks for a sharper render (zoom / retina); tall pages are capped by height and pixel count
        w = request.args.get("w", type=int) or 640
        w = max(320, min(w, 2560))
        png_bytes = _render_page_png(output_path, page_num, target_width_px=w,
                                     max_height_px=8000 if w <= 640 else 16000)
        if png_bytes is None:
            abort(404)
        from flask import Response
        return Response(png_bytes, mimetype="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})
    except Exception:
        abort(500)


@app.route("/preview-original/<job_id>/<int:page_num>")
def preview_original(job_id, page_num):
    """Render a page of the original annotated PDF as a PNG."""
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        abort(404)
    annotated_path = job.get("annotated_path")
    if not annotated_path or not os.path.exists(annotated_path):
        abort(404)
    try:
        png_bytes = _render_page_png(annotated_path, page_num)
        if png_bytes is None:
            abort(404)
        from flask import Response
        return Response(png_bytes, mimetype="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})
    except Exception:
        abort(500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

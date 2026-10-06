import os
import uuid
import threading
import time
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename
import dataclasses
import remapper as _R
from derivative import process_derivative
from remapper import process_pdfs, write_pdf, AnnotationInfo, ANNOT_FREETEXT, ANNOT_SQUARE
import boxsnap
from placer import place_references

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024  # 400MB total per upload (large design-deck PDFs)

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


def friendly_error(e):
    """Turn a processing exception into something a non-developer can act on."""
    msg = str(e)
    low = msg.lower()
    if isinstance(e, MemoryError) or "out of memory" in low or "cannot allocate" in low:
        return "The server ran out of memory on these files. Try smaller or compressed PDFs."
    if "password" in low or "encrypted" in low or "authenticate" in low:
        return "One of the PDFs is password-protected. Remove the password and try again."
    if "no annotations" in low:
        return msg
    if "cannot open" in low or "failed to open" in low or "not a pdf" in low or "format error" in low or "broken" in low or "no objects found" in low:
        return "One of the files couldn't be read as a PDF. It may be damaged; try re-exporting it."
    return "Something went wrong while processing these files (" + msg[:160] + "). Try again, or send the PDFs over so it can be looked at."



@app.errorhandler(413)
def _too_large(_e):
    return jsonify(error="Those files are too large to upload together (limit 400 MB combined). "
                         "Try compressing the PDFs or splitting them."), 413


@app.route("/")
def index():
    return render_template("index.html")


def _shape(path):
    """(page count, median width/height) of a PDF; width is the page without a Carryover sidebar."""
    import statistics
    import pymupdf
    d = pymupdf.open(path)
    try:
        ratios, widths = [], []
        for pg in d:
            w = pg.rect.width
            try:
                v = d.xref_get_key(pg.xref, "CPOrigW")[1]
                if v and v != "null":
                    w = float(v)
            except Exception:
                pass
            ratios.append(w / max(1.0, pg.rect.height))
            widths.append(w)
        return (len(d), (statistics.median(ratios) if ratios else 1.0),
                (statistics.median(widths) if widths else 1.0))
    finally:
        d.close()


def detect_relationship(old_path, new_path):
    """Is the new PDF an updated version of the annotated one, or a different (derivative) asset?
    Returns (mode, reason). A revision keeps roughly the same number of pages; a derivative is
    usually much shorter or longer, or a different page shape."""
    try:
        n_old, r_old, w_old = _shape(old_path)
        n_new, r_new, w_new = _shape(new_path)
    except Exception:
        return "update", "Could not compare the files, so they were treated as versions of the same document."
    ratio = min(n_old, n_new) / max(n_old, n_new, 1)
    if ratio < 0.6 and abs(n_old - n_new) >= 3:
        return "derivative", f"Page counts differ a lot ({n_old} vs {n_new})."
    # Same page count and same width: a taller or shorter page just means content was added or removed
    # (long single-page layouts), so it is still a version of the same document.
    same_width = abs(w_old - w_new) / max(w_old, 1e-6) < 0.05
    if not (n_old == n_new and same_width) and abs(r_old - r_new) / max(r_old, 1e-6) > 0.25:
        return "derivative", "The page shape is different."
    return "update", f"Same kind of document ({n_old} vs {n_new} pages)."


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

    request_mode = request.form.get("mode", "auto")   # auto | update | derivative
    job_id = str(uuid.uuid4())
    annotated_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_annotated.pdf")
    new_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_new.pdf")
    output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_output.pdf")

    annotated_file.save(annotated_path)
    new_file.save(new_path)

    jobs[job_id] = {"status": "processing", "percent": 0, "progress": "Uploaded - starting…"}

    def run_job():
        try:
            def prog(pct, text):
                j = jobs.get(job_id)
                if j and j.get("cancelled"):
                    raise Cancelled()
                if j and j.get("status") == "processing":
                    j["percent"] = round(pct, 1)
                    j["progress"] = text
            used, reason, auto = request_mode, "You switched to this.", False
            if request_mode == "auto":
                auto = True
                used, reason = detect_relationship(annotated_path, new_path)
                prog(1, "Comparing the two files…")
            jobs[job_id].update(mode_used=used, mode_reason=reason, mode_auto=auto)   # shown while processing
            fn = process_derivative if used == "derivative" else process_pdfs
            results = fn(annotated_path, new_path, output_path, prog)
            _free_pdf_cache()
            if _cancelled(job_id):
                raise Cancelled()
            if auto and used == "update" and results.get("total", 0) >= 5 \
                    and results.get("moved", 0) / results["total"] < 0.2:
                # Looked like a revision but almost nothing lined up: try it as a derivative and keep the better one.
                prog(1, "Few callouts lined up, trying a different approach…")
                alt = output_path[:-4] + "_d.pdf"
                alt_res = process_derivative(annotated_path, new_path, alt, prog)
                if alt_res.get("moved", 0) + alt_res.get("needs_look", 0) > results.get("moved", 0) + results.get("needs_look", 0):
                    for suffix in ("", "_preview"):
                        src, dst = alt[:-4] + suffix + ".pdf", output_path[:-4] + suffix + ".pdf"
                        if os.path.exists(src):
                            os.replace(src, dst)
                    alt_res["_preview_path"] = output_path[:-4] + "_preview.pdf"
                    results, used = alt_res, "derivative"
                    reason = "Few callouts lined up as an updated version, so they were matched by wording instead."
            results["mode_used"], results["mode_reason"], results["mode_auto"] = used, reason, auto
            if not results.get("total"):
                raise ValueError(
                    "No annotations were found in the first PDF. Upload the reviewer-annotated PDF (with live "
                    "comment/callout annotations) or an export from Carryover. Exports made before this "
                    "update have flattened pins and can't be re-used - re-run them from the original.")
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
        except Cancelled:
            jobs[job_id] = {"status": "cancelled"}
            for p in [annotated_path, new_path]:
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception as e:
            import traceback
            jobs[job_id] = {"status": "error", "message": friendly_error(e), "trace": traceback.format_exc()}
            for p in [annotated_path, new_path]:
                try:
                    os.remove(p)
                except Exception:
                    pass

    threading.Thread(target=run_job).start()
    return jsonify({"job_id": job_id})


class Cancelled(Exception):
    pass


def _cancelled(job_id):
    return bool((jobs.get(job_id) or {}).get("cancelled"))


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    job = jobs.get(job_id)
    if job and job.get("status") == "processing":
        job["cancelled"] = True
    return jsonify(ok=True)


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
        if job and job.get("cancelled"):
            raise Cancelled()
        if job and job.get("status") == "processing":
            job["progress"] = f"Reading page {done} of {total}…"
            job["percent"] = round(85 * done / max(1, total), 1)

    def run_job():
        try:
            results = place_references(doc_path, pdf_path, output_path, progress)
            if _cancelled(job_id):
                raise Cancelled()
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
        except Cancelled:
            jobs[job_id] = {"status": "cancelled"}
            for p in [doc_path, pdf_path]:
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception as e:
            import traceback
            jobs[job_id] = {"status": "error", "message": friendly_error(e), "trace": traceback.format_exc()}

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


def _effective_annots(all_annots, pdf_path, edits, new_boxes, overrides):
    """The annotation list for this export: stored annotations (with any text edits, on copies so a later
    export can undo them) plus the boxes the user drew in the review page. New pins go into `overrides`
    so they are drawn exactly where they were placed."""
    eff = [dataclasses.replace(a, content=edits[a.index]) if a.index in edits else a for a in all_annots]
    if not new_boxes:
        return eff
    doc = fitz.open(pdf_path)
    try:
        for nb in new_boxes:
            try:
                pn = int(nb["page"]) - 1
                bi, pi = int(nb["index"]), int(nb["pin_index"])
                has_box = nb.get("box") is not None            # a pin added on its own has no box
                fx, fy, fw, fh = [float(v) for v in nb["box"]] if has_box else (0.0, 0.0, 0.0, 0.0)
                px, py = [float(v) for v in nb["pin"]]
                page = doc[pn]
            except (KeyError, ValueError, TypeError, IndexError):
                continue
            pw, ph = page.rect.width, page.rect.height
            tot_w = pw + _R.SIDEBAR_BASE * _R._scale(pw)
            fx, fy = max(0.0, min(1.0, fx)), max(0.0, min(1.0, fy))
            box = fitz.Rect(fx * tot_w, fy * ph, min(pw, (fx + fw) * tot_w), min(ph, (fy + fh) * ph + 0))
            tx, ty = px * tot_w, py * ph
            content = str(nb.get("content") or "")[:5000]
            sq = AnnotationInfo(index=bi, page_num=pn, annot_type=(ANNOT_SQUARE, "Square"), rect=fitz.Rect(box),
                                content="", vertices=[], flags=0, colors={}, border={})
            sq.new_rect = fitz.Rect(box); sq.new_page_num = pn; sq.status = "moved"; sq.matched = True
            pin_rect = fitz.Rect(tx - 6, ty - 6, tx + 6, ty + 6)
            pin = AnnotationInfo(index=pi, page_num=pn, annot_type=(ANNOT_FREETEXT, "FreeText"), rect=pin_rect,
                                 content=content, vertices=[(tx, ty), (tx + 30, ty - 20)], flags=0, colors={}, border={})
            pin.tip_point = fitz.Point(tx, ty)
            pin.new_rect = pin_rect; pin.new_vertices = [(tx, ty), (tx + 30, ty - 20)]
            pin.new_page_num = pn; pin.status = "moved"; pin.matched = True
            eff += [sq, pin] if has_box else [pin]
            overrides.setdefault(pi, (px, py))
    finally:
        doc.close()
    return eff


@app.route("/tighten/<job_id>", methods=["POST"])
def tighten_box(job_id):
    """Shrink a roughly drawn box to the content inside it. Body: {page, rect:[x,y,w,h]} as fractions of the output page."""
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Job not ready"}), 404
    data = request.get_json(silent=True) or {}
    path = job.get("new_pdf_path")
    try:
        pn = int(data["page"])
        fx, fy, fw, fh = [float(v) for v in data["rect"]]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Bad request"}), 400
    if not path or not os.path.exists(path):
        return jsonify({"error": "Source PDF no longer available"}), 500
    try:
        doc = fitz.open(path)
        page = doc[pn - 1]
        pw, ph = page.rect.width, page.rect.height
        doc.close()
        tot_w = pw + _R.SIDEBAR_BASE * _R._scale(pw)
        rect = fitz.Rect(fx * tot_w, fy * ph, (fx + fw) * tot_w, (fy + fh) * ph)
        tight, changed = boxsnap.tighten(path, pn, rect)
        return jsonify({"rect": [tight.x0 / tot_w, tight.y0 / ph, tight.width / tot_w, tight.height / ph],
                        "changed": bool(changed)})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


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
            if isinstance(v, (list, tuple)) and len(v) == 2:
                sides[int(k)] = (float(v[0]), float(v[1]))     # ring dragged to a free spot: offset in ring radii
            else:
                sides[int(k)] = int(v) % 4      # quarter turns: 0 right, 1 below, 2 left, 3 above
        except (ValueError, TypeError):
            continue
    overrides = {}
    for k, v in (data.get("positions") or {}).items():
        try:
            idx = int(k)
            fx, fy = float(v[0]), float(v[1])
        except (ValueError, TypeError, IndexError):
            continue
        if fx == fx and fy == fy:                       # not NaN; a pin point dragged a hair past the edge is clamped
            overrides[idx] = (min(1.0, max(0.0, fx)), min(1.0, max(0.0, fy)))

    all_annots = job.get("_all_annots")
    new_pdf_path = job.get("new_pdf_path") or job.get("annotated_path")
    output_path = job.get("output_path")

    def _fracs(v, n):
        try:
            vals = [float(x) for x in v]
        except (ValueError, TypeError):
            return None
        return vals if len(vals) == n and all(x == x for x in vals) else None
    # Edited boxes: { "<annotation index>": [x, y, w, h] } as fractions of the output page
    boxes = {}
    for k, v in (data.get("boxes") or {}).items():
        f = _fracs(v, 4)
        if f and str(k).lstrip("-").isdigit():
            boxes[int(k)] = tuple(f)
    # Edited reference text: { "<annotation index>": "text" }
    edits = {}
    for k, v in (data.get("edits") or {}).items():
        if str(k).lstrip("-").isdigit() and isinstance(v, str):
            edits[int(k)] = v[:5000]
    new_boxes = data.get("new_boxes") or []

    if not all_annots or not new_pdf_path or not output_path:
        return jsonify({"error": "Job data missing — please re-process"}), 500

    if not os.path.exists(new_pdf_path):
        return jsonify({"error": "Source PDF no longer available — please re-process"}), 500

    try:
        eff = _effective_annots(all_annots, new_pdf_path, edits, new_boxes, overrides)
        tracking = data.get("tracking")
        tracking_bytes = None
        if isinstance(tracking, dict):
            import json as _json
            tracking_bytes = _json.dumps(tracking, separators=(",", ":")).encode("utf-8")[:2_000_000]
        write_pdf(eff, new_pdf_path, output_path, skip_indices=deleted, overrides=overrides, sides=sides, boxes=boxes,
                  tracking=tracking_bytes)
        _free_pdf_cache()
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


MAX_PREVIEW_PIXELS = 14_000_000    # keeps one render's memory bounded on very tall pages
# The review screen asks for ~90 page images at once; on a small server (512 MB) rendering many big pages in
# parallel is what runs it out of memory, so only a couple render at a time and the rest wait their turn.
_render_gate = threading.BoundedSemaphore(2)


def _free_pdf_cache():
    """MuPDF keeps every decoded image it has drawn (hundreds of MB on big PDFs). Drop that cache so
    the server's memory stays flat on a small instance."""
    try:
        fitz.TOOLS.store_shrink(100)
    except Exception:
        pass


def _render_page_png(pdf_path, page_num, target_width_px=640, max_height_px=8000):
    """Render a PDF page scaled so its width fits target_width_px."""
    with _render_gate:
        return _render_page_png_locked(pdf_path, page_num, target_width_px, max_height_px)


def _render_page_png_locked(pdf_path, page_num, target_width_px, max_height_px):
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
        data = pix.tobytes("png")
        del pix
        return data
    finally:
        doc.close()
        _free_pdf_cache()


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
        w = max(320, min(w, 1920))
        png_bytes = _render_page_png(output_path, page_num, target_width_px=w,
                                     max_height_px=8000 if w <= 640 else 12000)
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

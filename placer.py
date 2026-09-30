"""Place references from a Word document onto a first-time layout PDF.

Input : a .docx reference list + the layout PDF (text layer or flat image)
Output: the same result structure Carryover's remap flow produces, so the review
        screen, pin editing and export work unchanged.

How each reference row finds its spot on the page:
  1. Claim column filled in  -> fuzzy-search that copy on the page(s).
  2. No claim, has a number  -> pair the Nth row for reference "n" with the Nth
     superscript "n" in reading order. A count mismatch is flagged, never guessed.
  3. Neither works           -> the row is kept as an unplaced pin the user drags.
Pages without a text layer are read with OCR (best effort, always flagged).
"""
import io
import re
import difflib
from collections import defaultdict

import fitz

from remapper import (AnnotationInfo, ANNOT_FREETEXT, _finalize)

OCR_SCALE = 2
STRIP_PX = 3000            # OCR tall pages in strips this many pixels high
MIN_TEXT_WORDS = 25        # fewer words than this on a page = treat as an image


# ─────────────────────────────── Word document ───────────────────────────────

_HEADER_KEYS = {
    "num":   ("#", "no", "no.", "ref", "ref #", "ref no", "number", "superscript", "sup"),
    "claim": ("claim", "layout", "copy", "anchor", "text on page", "statement"),
    "ref":   ("reference", "source", "location", "citation"),
    "page":  ("page", "pg"),
}


def _clean(s):
    return re.sub(r"[ \t]+", " ", (s or "").replace(" ", " ")).strip()


def _classify_header(h):
    h = h.lower().strip()
    if h in _HEADER_KEYS["num"]:
        return "num"
    for key in ("claim", "ref", "page"):
        if any(k in h for k in _HEADER_KEYS[key]):
            return key
    if "num" in h or "superscript" in h:
        return "num"
    return None


def parse_docx(path):
    """Return [{'num': int|None, 'claim': str, 'ref': str, 'page': int|None}, ...] in document order."""
    from docx import Document
    doc = Document(path)
    rows = []
    for table in doc.tables:
        if not table.rows:
            continue
        head = [_classify_header(c.text) for c in table.rows[0].cells]
        has_header = "ref" in head or ("num" in head and len(set(head) - {None}) >= 2)
        if has_header:
            cols = head
            body = table.rows[1:]
        else:                                       # headerless: assume  # | claim? | reference [| page]
            n = len(table.rows[0].cells)
            cols = {1: ["ref"], 2: ["num", "ref"], 3: ["num", "claim", "ref"]}.get(n, ["num", "claim", "ref", "page"])
            body = table.rows
        for r in body:
            vals = {}
            for kind, cell in zip(cols, r.cells):
                if kind and kind not in vals:
                    vals[kind] = cell.text.strip()
            ref = _clean_multiline(vals.get("ref", ""))
            if not ref:
                continue
            rows.append(_row(vals.get("num"), vals.get("claim"), ref, vals.get("page")))
        if rows:
            return rows
    # No table: numbered paragraphs  "3. Maggi 2023/p2/..."  or  "3 - text"
    for p in doc.paragraphs:
        m = re.match(r"^\s*(\d{1,3})[\.\)\-–:]\s*(.+)$", p.text.strip())
        if m:
            rows.append(_row(m.group(1), "", m.group(2), None))
    return rows


def _clean_multiline(s):
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in re.split(r"[\r\n]+", s or "")]
    return "\n".join(l for l in lines if l)


def _row(num, claim, ref, page):
    n = None
    if num:
        m = re.search(r"\d+", str(num))
        n = int(m.group()) if m else None
    p = None
    if page:
        m = re.search(r"\d+", str(page))
        p = int(m.group()) if m else None
    return {"num": n, "claim": _clean(claim or ""), "ref": ref, "page": p}


# ───────────────────────────── page text (layer or OCR) ──────────────────────

def _norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


def _page_text_words(page):
    """Words on the page that are really part of the layout, i.e. not text drawn inside an
    annotation (callouts carry their own text, which would otherwise look like a text layer)."""
    boxes = [fitz.Rect(a.rect) for a in page.annots()]
    out = []
    for w in page.get_text("words"):
        r = fitz.Rect(w[:4])
        if not any(r.intersects(b) for b in boxes):
            out.append(w)
    return out


def _has_text_layer(page):
    return len(_page_text_words(page)) >= MIN_TEXT_WORDS


def _ocr_words(page):
    """OCR one page in strips. Returns [(text, x0, y0, x1, y1, order_key)] in PDF points."""
    import pytesseract
    from PIL import Image
    pix = page.get_pixmap(matrix=fitz.Matrix(OCR_SCALE, OCR_SCALE), alpha=False, annots=False)
    im = Image.open(io.BytesIO(pix.tobytes("png")))
    W, H = im.size
    words, seq = [], 0
    y = 0
    while y < H:
        y2 = min(H, y + STRIP_PX)
        strip = im.crop((0, y, W, y2))
        d = pytesseract.image_to_data(strip, output_type=pytesseract.Output.DICT, config="--psm 11")
        for i, t in enumerate(d["text"]):
            t = t.strip()
            if t and float(d["conf"][i]) > 30:
                x0 = d["left"][i] / OCR_SCALE
                y0 = (d["top"][i] + y) / OCR_SCALE
                x1 = (d["left"][i] + d["width"][i]) / OCR_SCALE
                y1 = (d["top"][i] + y + d["height"][i]) / OCR_SCALE
                words.append((t, x0, y0, x1, y1, (y, d["block_num"][i], d["par_num"][i], d["line_num"][i], x0)))
        y = y2
    words.sort(key=lambda w: w[5])
    return words


def load_pages(pdf_path, page_filter=None, progress=None):
    """{page_index: {'words': [...], 'ocr': bool, 'doc_page': fitz.Page-size info}}"""
    doc = fitz.open(pdf_path)
    pages = {}
    todo = [i for i in range(len(doc)) if page_filter is None or i in page_filter]
    for k, i in enumerate(todo):
        page = doc[i]
        if _has_text_layer(page):
            ws = [(w[4], w[0], w[1], w[2], w[3], (w[5], w[6], w[7])) for w in _page_text_words(page)]
            ws.sort(key=lambda w: w[5])
            pages[i] = {"words": ws, "ocr": False}
        else:
            pages[i] = {"words": _ocr_words(page), "ocr": True}
        if progress:
            progress(k + 1, len(todo))
    doc.close()
    return pages


# ─────────────────────────────── superscripts ────────────────────────────────

_SUP_RE = re.compile(r"^[\d,\s\-–—]+$")


def _expand_numbers(txt):
    out = []
    for part in re.split(r"[,\s]+", txt.strip()):
        if not part:
            continue
        m = re.match(r"^(\d+)[\-–—](\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a <= 12:
                out.extend(range(a, b + 1))
                continue
        if part.isdigit():
            out.append(int(part))
    return out


def find_superscripts(pdf_path, pages):
    """Superscript reference numbers in reading order.
    Returns [{'num', 'page', 'x', 'y', 'rect', 'ocr': bool}] (x,y = spot the pin points at)."""
    doc = fitz.open(pdf_path)
    found = []
    for pi in sorted(pages):
        page = doc[pi]
        if not pages[pi]["ocr"]:
            found.extend(_sups_from_text_layer(page, pi))
        else:
            found.extend(_sups_from_ocr(pages[pi]["words"], pi))
    doc.close()
    return found


def _sups_from_text_layer(page, pi):
    out = []
    boxes = [fitz.Rect(a.rect) for a in page.annots()]
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            if any(fitz.Rect(ln["bbox"]).intersects(bx) for bx in boxes):
                continue                       # text drawn inside an annotation, not layout copy
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if len(spans) < 2:
                continue
            body = max((s["size"] for s in spans if re.search(r"[A-Za-z]", s["text"])), default=0)
            if not body:
                continue
            for k, s in enumerate(spans):
                t = s["text"].strip()
                if k == 0 or not _SUP_RE.match(t) or not re.search(r"\d", t):
                    continue
                small = s["size"] <= body * 0.9
                raised = bool(s["flags"] & 1) or (s["bbox"][3] < ln["bbox"][3] - body * 0.25)
                prev = spans[k - 1]["text"].rstrip()
                if not (small and raised) or not prev:
                    continue
                r = fitz.Rect(s["bbox"])
                for n in _expand_numbers(t):
                    out.append({"num": n, "page": pi, "x": r.x0, "y": (r.y0 + r.y1) / 2,
                                "rect": r, "ocr": False, "key": (pi, round(ln["bbox"][1] / 4), r.x0)})
    out.sort(key=lambda o: (o["page"], round(o["rect"].y0 / 8), o["rect"].x0))
    return out


_OCR_SUP = re.compile(r"^(.*?[A-Za-z\)\.\,;:])(\d{1,2}(?:[,\-–]\d{1,2})*)[\.\,;]?$")


def _sups_from_ocr(words, pi):
    """OCR can't tell a raised '1' from a digit: treat trailing digits glued to a word as candidates."""
    out = []
    for (t, x0, y0, x1, y1, _k) in words:
        m = _OCR_SUP.match(t)
        if not m or len(m.group(1)) < 3:
            continue
        pre, digits = m.group(1), m.group(2)
        frac = len(pre) / max(1, len(t))
        sx = x0 + (x1 - x0) * frac
        for n in _expand_numbers(digits):
            out.append({"num": n, "page": pi, "x": sx, "y": y0 + (y1 - y0) * 0.3,
                        "rect": fitz.Rect(sx, y0, x1, y1), "ocr": True})
    out.sort(key=lambda o: (o["page"], round(o["rect"].y0 / 8), o["rect"].x0))
    return out


# ───────────────────────────────── claim search ──────────────────────────────

def locate_claim(claim, pages, page_hint=None):
    """Fuzzy-find `claim` in page words. Returns dict(score, page, rect, ambiguous, ocr) or None."""
    at = _norm(claim).split()
    if len(at) < 2:
        return None
    n = len(at)
    target = " ".join(at)
    tset = set(at)
    cands = []
    scope = [page_hint] if page_hint is not None and page_hint in pages else list(pages)
    for pi in scope:
        ws = pages[pi]["words"]
        toks = [_norm(w[0]) for w in ws]
        for i in range(0, max(1, len(ws) - n + 1)):
            win = toks[i:i + n]
            if len(win) < n:
                continue
            if len(tset & set(win)) < max(1, int(len(tset) * 0.5)):      # cheap pre-filter
                continue
            s = difflib.SequenceMatcher(None, target, " ".join(win)).ratio()
            if s >= 0.55:
                seg = ws[i:i + n]
                r = fitz.Rect(min(w[1] for w in seg), min(w[2] for w in seg),
                              max(w[3] for w in seg), max(w[4] for w in seg))
                cands.append((s, pi, r, pages[pi]["ocr"], seg[-1]))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    best = cands[0]
    rivals = [c for c in cands[1:]
              if c[1] != best[1] or abs(c[2].y0 - best[2].y0) > 30 or abs(c[2].x0 - best[2].x0) > 60]
    amb = bool(rivals) and rivals[0][0] >= best[0] - 0.05 and best[0] > 0.6
    last = best[4]
    return {"score": best[0], "page": best[1], "rect": best[2], "ambiguous": amb,
            "ocr": best[3], "x": last[3], "y": (last[2] + last[4]) / 2}


# ───────────────────────────────── main entry ────────────────────────────────

def place_references(docx_path, pdf_path, output_path, progress=None):
    rows = parse_docx(docx_path)
    if not rows:
        raise ValueError("No references found in the Word document. Use a table with a "
                         "'Reference / source location' column (see the template).")
    src = fitz.open(pdf_path)
    npages = len(src)
    src.close()

    # Which pages need to be read? Every page, unless every row names its page.
    hinted = {r["page"] - 1 for r in rows if r["page"]}
    all_hinted = all(r["page"] for r in rows)
    page_filter = {p for p in hinted if 0 <= p < npages} if all_hinted else None
    pages = load_pages(pdf_path, page_filter, progress)

    sups = find_superscripts(pdf_path, pages)
    sup_by_num = defaultdict(list)
    for s in sups:
        sup_by_num[s["num"]].append(s)
    rows_by_num = defaultdict(list)
    for idx, r in enumerate(rows):
        if r["num"] is not None and not r["claim"]:
            rows_by_num[r["num"]].append(idx)

    placed = {}            # row idx -> dict(page, x, y, status, conf, method, note)

    # 1) claim rows
    for idx, r in enumerate(rows):
        if not r["claim"]:
            continue
        hint = (r["page"] - 1) if r["page"] else None
        hit = locate_claim(r["claim"], pages, hint)
        if hit is None and hint is not None:
            hit = locate_claim(r["claim"], pages, None)          # page hint may be wrong
        if hit is None:
            continue
        good = hit["score"] >= (0.9 if hit["ocr"] else 0.8) and not hit["ambiguous"]
        how = " (read from an image with OCR)" if hit["ocr"] else ""
        if hit["ambiguous"]:
            note = "The same wording appears more than once — check this is the right one"
        elif good:
            note = f"Matched the copy{how}, score {hit['score']:.2f}"
        else:
            note = f"Approximate match{how}, score {hit['score']:.2f} — check the spot"
        ok = good
        placed[idx] = dict(page=hit["page"], x=hit["x"], y=hit["y"], status="moved" if ok else "needs_look",
                           conf=hit["score"], method="text", note=note, rect=hit["rect"])

    # 2) superscript pairing (rows with a number and no claim text)
    for num, idxs in rows_by_num.items():
        occ = sup_by_num.get(num, [])
        if not occ:
            continue
        exact = len(occ) == len(idxs)
        for k, idx in enumerate(idxs):
            if k >= len(occ):
                break
            s = occ[k]
            via_ocr = s["ocr"]
            if exact and not via_ocr:
                status, conf, note = "moved", 0.9, f"Paired with superscript {num} (#{k + 1} of {len(occ)} on the document)"
            else:
                status, conf = "needs_look", 0.45
                note = (f"Reference {num} has {len(idxs)} row(s) but {len(occ)} superscript(s) were found — "
                        f"pairing is by order, so check this one" if not exact
                        else f"Superscript {num} was read from an image with OCR — check the spot")
            placed[idx] = dict(page=s["page"], x=s["x"], y=s["y"], status=status, conf=conf,
                               method="text", note=note, rect=s["rect"])

    # 3) build annotations
    all_annots = []
    unplaced_n = defaultdict(int)
    results = {"total": 0, "matched": 0, "unmatched": 0, "moved": 0, "needs_look": 0,
               "content_removed": 0, "skipped": [], "annotations": [], "mode": "add"}
    doc = fitz.open(pdf_path)
    for idx, r in enumerate(rows):
        p = placed.get(idx)
        if p is None:
            pg = min(max((r["page"] or 1) - 1, 0), npages - 1)
            k = unplaced_n[pg]; unplaced_n[pg] += 1
            p = dict(page=pg, x=26.0, y=40.0 + k * 34.0, status="needs_look", conf=0.0, method="position",
                     note="Couldn't find where this goes — drag the pin to its spot", rect=None)
            matched = False
        else:
            matched = True
        x, y = p["x"], p["y"]
        rect = fitz.Rect(x - 6, y - 6, x + 6, y + 6)
        info = AnnotationInfo(
            index=idx, page_num=p["page"], annot_type=(ANNOT_FREETEXT, "FreeText"),
            rect=rect, content=r["ref"], vertices=[(x, y), (x + 30, y - 20)], flags=0,
            colors={}, border={},
        )
        info.tip_point = fitz.Point(x, y)
        info.matched = matched
        info.match_confidence = p["conf"]
        info.match_method = p["method"]
        info.status = p["status"]
        info.new_rect = rect
        info.new_vertices = [(x, y), (x + 30, y - 20)]
        info.new_page_num = p["page"]
        info.match_note = p["note"]
        info.fingerprint_text = r["claim"]
        all_annots.append(info)

        results["total"] += 1
        results["matched" if matched else "unmatched"] += 1
        results["moved" if p["status"] == "moved" else "needs_look"] += 1
        results["annotations"].append({
            "index": idx, "page": p["page"] + 1, "type": "FreeText",
            "content": r["ref"][:80] + ("..." if len(r["ref"]) > 80 else ""),
            "author": "", "status": p["status"], "confidence": f"{p['conf']:.0%}",
            "method": p["method"], "note": p["note"], "parent_index": None,
            "ref_num": r["num"],
        })
    doc.close()
    results["ocr_pages"] = sum(1 for v in pages.values() if v["ocr"])
    return _finalize(results, all_annots, None, pdf_path, output_path)

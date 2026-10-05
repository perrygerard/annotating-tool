"""Derivative carry-over: find where each annotated claim of a SOURCE asset lives in a
DERIVATIVE piece (different length, layout and page size), instead of assuming the
derivative is a revision of the source.

For every callout on the source we take the text it points at (live text, or OCR when the
source page is a flat image), then search the derivative's text for it. Results are graded:
  found       - the claim text is in the derivative (pin placed on it)
  check       - partial / short match, a person should look
  not_in_piece- nothing close: the claim was cut from the derivative
"""
import re
from difflib import get_close_matches

import pymupdf as fitz

STOP = {"the", "and", "for", "are", "was", "with", "this", "that", "from", "have", "has", "been",
        "not", "but", "also", "more", "were", "all", "any", "can", "may", "use", "its", "per"}


def norm_tok(t):
    return re.sub(r"[^a-z0-9%+.\-]", "", t.lower()).strip(".-")


def tokens(words):
    out = []
    for w in words:
        t = norm_tok(w)
        if len(t) >= 3 and t not in STOP:
            out.append(t)
    return out


# ── source side: the claim each callout points at ────────────────────────────

def callouts(doc):
    """[{page, tip:(x,y), content, rect}] for every FreeText callout (tip from /CL)."""
    out = []
    for pn, page in enumerate(doc):
        ph = page.mediabox.height
        for a in page.annots() or []:
            if a.type[0] != 2:
                continue
            cl = doc.xref_get_key(a.xref, "CL")[1] or ""
            nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", cl)]
            if len(nums) < 4:
                continue
            tip = (nums[0] - page.mediabox.x0, ph - (nums[1] - page.mediabox.y0))
            out.append({"page": pn, "tip": tip, "content": a.info.get("content", ""),
                        "rect": fitz.Rect(a.rect), "xref": a.xref})
    return out


def squares(doc):
    out = []
    for pn, page in enumerate(doc):
        for a in page.annots() or []:
            if a.type[0] == 4:
                out.append({"page": pn, "rect": fitz.Rect(a.rect)})
    return out


def _dist(r, p):
    dx = max(r[0] - p[0], 0, p[0] - r[2])
    dy = max(r[1] - p[1], 0, p[1] - r[3])
    return (dx * dx + dy * dy) ** 0.5


def claim_words(words, tip, boxes=(), max_area=1456 * 1200 * 0.35):
    """words: [(text,x0,y0,x1,y1,linekey)] of the source page, WITHOUT annotation text.
    Returns the words of the OCR lines at the tip (or inside the box the tip is in)."""
    # prefer the highlight box the tip sits inside, if it is a modest size
    inside = [b for b in boxes if b.contains(fitz.Point(*tip)) or _dist(tuple(b), tip) < 4]
    inside.sort(key=lambda b: b.get_area())
    if inside and inside[0].get_area() < max_area:
        r = inside[0]
        ws = [w for w in words if fitz.Rect(w[1:5]).intersects(r)]
        if len(ws) >= 2:
            return ws, "box"
    for rad in (30, 60, 110, 170):
        seed = [w for w in words if _dist(w[1:5], tip) <= rad]
        if len(seed) >= 3:
            keys = {tuple(w[5]) for w in seed}
            ws = [w for w in words if tuple(w[5]) in keys]
            return ws, "tip"
    return [], "none"


# ── derivative side ──────────────────────────────────────────────────────────

class Index:
    def __init__(self, doc):
        self.pages = []
        self.vocab = set()
        for pn, page in enumerate(doc):
            ws = []
            for w in page.get_text("words"):
                t = norm_tok(w[4])
                if len(t) >= 3 and t not in STOP:
                    ws.append((t, fitz.Rect(w[:4])))
                    self.vocab.add(t)
            self.pages.append(ws)
        from collections import Counter
        c = Counter(t for ws in self.pages for t, _ in ws)
        # words that appear all over the piece (brand, product, 'week', units...) prove nothing alone
        self.generic = {t for t, _ in c.most_common(12)}

    def fuzzy(self, tok):
        if tok in self.vocab:
            return tok
        if len(tok) >= 5:
            m = get_close_matches(tok, self.vocab, n=1, cutoff=0.84)
            if m:
                return m[0]
        return None


def locate(idx, claim_toks):
    """Where does the claim's wording appear in the derivative?
    Order-aware: runs of >=2 consecutive claim tokens that also run consecutively in a derivative
    page (so common words alone, like 'EYLEA HD week 96', do not count).
    Returns (coverage 0-1 of the claim's informative tokens, page, rect)."""
    from difflib import SequenceMatcher
    if not claim_toks:
        return 0.0, None, None
    mapped = [idx.fuzzy(t) or t for t in claim_toks]
    info = [t for t in mapped if t not in idx.generic]
    if not info:
        return 0.0, None, None
    best = (0.0, None, None)
    for pn, ws in enumerate(idx.pages):
        if len(ws) < 2:
            continue
        toks = [t for t, _ in ws]
        sm = SequenceMatcher(None, mapped, toks, autojunk=False)
        got, rects = 0, []
        for blk in sm.get_matching_blocks():
            if blk.size < 2:
                continue
            seg = mapped[blk.a:blk.a + blk.size]
            n_inf = sum(1 for t in seg if t not in idx.generic)
            if n_inf < 2 and blk.size < 4:
                continue
            got += n_inf
            rects += [ws[blk.b + i][1] for i in range(blk.size)]
        cov = got / len(info)
        if cov > best[0] and rects:
            r = fitz.Rect(rects[0])
            for q in rects[1:]:
                r |= q
            best = (cov, pn, r)
    return best


def grade(cov, n_tokens):
    if n_tokens < 3:
        return "check"          # too little text to trust
    if cov >= 0.6:
        return "found"
    if cov >= 0.25:
        return "check"      # same claim, wording changed (or OCR noise): a person confirms
    return "not_in_piece"


# ── app entry point ──────────────────────────────────────────────────────────

def process_derivative(src_path, deriv_path, output_path, progress=None):
    """Carry the callouts of an annotated SOURCE over to a DERIVATIVE piece by finding each claim's
    wording in it. Same result shape as remapper.process_pdfs, so the review screen is shared."""
    import remapper as R
    import placer
    prog = progress or (lambda p, t: None)
    src = fitz.open(src_path)
    clean = fitz.open(src_path)
    for pg in clean:
        for a in list(pg.annots() or []):
            pg.delete_annot(a)
    deriv = fitz.open(deriv_path)
    prog(3, "Reading the new piece")
    idx = Index(deriv)
    cs = callouts(src)
    boxes = {}
    for pn, page in enumerate(src):
        for a in page.annots() or []:
            if a.type[0] in (4, 8):
                boxes.setdefault(pn, []).append(fitz.Rect(a.rect))

    results = {"total": 0, "matched": 0, "unmatched": 0, "moved": 0, "needs_look": 0,
               "content_removed": 0, "skipped": [], "annotations": [], "derivative": True}
    words_cache = {}

    def page_words(pn):
        if pn not in words_cache:
            page = clean[pn]
            live = page.get_text("words")
            if len(live) >= 15:
                ws = [(w[4], w[0], w[1], w[2], w[3], (w[5], w[6], 0)) for w in live]
            else:
                ws = placer._ocr_words(page)
            words_cache[pn] = ws
        return words_cache[pn]

    n_src, n_der = len(src), len(deriv)
    all_annots = []
    for k, c in enumerate(cs):
        prog(5 + 80 * k / max(1, len(cs)), f"Finding claim {k + 1} of {len(cs)} in the new piece")
        pn = c["page"]
        page = src[pn]
        pa = page.rect.get_area()
        ws, how = claim_words(page_words(pn), c["tip"], boxes.get(pn, []), max_area=0.35 * pa)
        toks = tokens([w[0] for w in ws])
        cov, fp, rect = locate(idx, toks)
        g = grade(cov, len(set(toks)))
        if fp is None and g != "not_in_piece":
            g = "not_in_piece"
        sa = c["rect"]
        info = R.AnnotationInfo(
            index=k, page_num=pn, annot_type=(R.ANNOT_FREETEXT, "FreeText"), rect=sa,
            content=c["content"], vertices=[c["tip"], ((sa.x0 + sa.x1) / 2, (sa.y0 + sa.y1) / 2)],
            flags=0, colors={}, border={}, tip_point=fitz.Point(*c["tip"]),
            fingerprint_text=" ".join(w[0] for w in ws)[:200])
        if g == "not_in_piece":
            # Nowhere to put it: park it at the scaled spot, flagged, and the review screen leaves it out
            # of the export until the reviewer places it.
            dp = round(pn * (n_der - 1) / max(1, n_src - 1))
            dpg = deriv[dp]
            sx, sy = dpg.rect.width / page.rect.width, dpg.rect.height / page.rect.height
            tx, ty = c["tip"][0] * sx, c["tip"][1] * sy
            info.new_page_num = dp
            info.new_rect = fitz.Rect(tx, ty - 6, tx + 12, ty + 6)
            info.new_vertices = [(tx, ty), (tx + 30, ty)]
            info.status, info.match_method, info.match_confidence = "content_removed", "absent", 0.0
            info.match_note = "Not found in this piece. Drag the pin onto the claim if it is there, or leave it removed."
            results["content_removed"] += 1
            results["unmatched"] += 1
        else:
            cy = (rect.y0 + rect.y1) / 2
            info.new_page_num = fp
            info.new_rect = fitz.Rect(rect)
            info.new_vertices = [(rect.x1, cy), (rect.x1 + 30, cy)]
            info.matched = True
            info.match_confidence = min(1.0, cov)
            info.match_method = "claim"
            if g == "found":
                info.status = "moved"
                info.match_note = "Claim wording found on this page."
                results["moved"] += 1
            else:
                info.status = "needs_look"
                info.match_note = "Probably the same claim with different wording. Check the pin is on the right line."
                results["needs_look"] += 1
            results["matched"] += 1
        results["total"] += 1
        all_annots.append(info)
        results["annotations"].append({
            "index": info.index, "page": pn + 1, "type": "FreeText",
            "content": info.content[:80] + ("..." if len(info.content) > 80 else ""),
            "author": info.author, "status": info.status,
            "confidence": f"{info.match_confidence:.0%}", "method": info.match_method,
            "note": info.match_note, "parent_index": None})
    src.close(); clean.close(); deriv.close()
    return R._finalize(results, all_annots, src_path, deriv_path, output_path, progress)

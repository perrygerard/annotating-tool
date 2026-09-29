"""
Tip-anchored visual matching for bitmap (image-only) pages.

For each annotation we take the point it actually references (a callout's arrow
tip, or the centre of a rectangle), cut a patch of the OLD page around that
point (rendered WITHOUT annotations), and find the same patch in the NEW page
with normalised cross-correlation.

Why this beats a page-wide shift estimate:
  * The patch contains the text/graphics the reviewer was pointing at, so the
    match is about the referenced content itself, not the margin box.
  * Every annotation gets its own shift, so multi-section edits are handled.
  * The correlation peak is a direct confidence signal: ~0.9+ means the content
    is present; a low peak means it was removed or rewritten.
"""

try:
    import numpy as np
    import cv2
    HAS_CV = True
except Exception:          # opencv/numpy not installed -> caller falls back
    HAS_CV = False

import fitz

TARGET_PX_W = 960          # work resolution (pixel width of rendered pages)
CONFIDENT = 0.75           # min correlation to trust a match
MARGIN = 0.08              # min gap to the runner-up (else ambiguous)
REMOVED_BELOW = 0.50       # below this the referenced content is treated as gone
MIN_TEXTURE = 8.0          # patch std-dev; flatter patches carry no signal


def _render_gray(page):
    """Render a page to grayscale WITHOUT annotations. Returns (array, scale)."""
    scale = min(1.0, TARGET_PX_W / page.rect.width) if page.rect.width > TARGET_PX_W else 1.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), annots=False,
                          alpha=False, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    return arr, scale


def anchor_for(info, ANNOT_FREETEXT, ANNOT_SQUARE):
    """The point in the OLD page this annotation refers to, or None."""
    if info.annot_type[0] == ANNOT_FREETEXT and info.vertices:
        return (info.vertices[0][0], info.vertices[0][1])       # arrow tip
    if info.annot_type[0] == ANNOT_SQUARE:
        r = info.rect
        return ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
    return None


def _match_one(old_g, old_s, new_g, new_s, old_pw, pt, x_drift_pt=20.0):
    """Match the patch around pt (old page points) inside new_g.
    Returns dict(score, second, tip_new=(x,y) in new-page points) or None if
    the patch is too flat to carry any signal."""
    hw = 0.135 * old_pw           # patch half width  (pts)
    hh = 0.047 * old_pw           # patch half height (pts)
    x, y = pt
    H, W = old_g.shape
    x0 = int(max(0, (x - hw) * old_s)); x1 = int(min(W, (x + hw) * old_s))
    y0 = int(max(0, (y - hh) * old_s)); y1 = int(min(H, (y + hh) * old_s))
    if x1 - x0 < 20 or y1 - y0 < 12:
        return None
    tpl = old_g[y0:y1, x0:x1]
    if float(tpl.std()) < MIN_TEXTURE:
        return None

    # New page is rendered to the same pixel width, so x maps 1:1 in pixels.
    drift = int(x_drift_pt * old_s)
    ex0 = max(0, x0 - drift); ex1 = min(new_g.shape[1], x1 + drift)
    strip = new_g[:, ex0:ex1]
    if strip.shape[0] < tpl.shape[0] or strip.shape[1] < tpl.shape[1]:
        return None

    res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(res)
    # runner-up: best score outside +-hh of the winner (uniqueness check)
    guard = max(4, int(hh * old_s))
    r2 = res.copy()
    r2[max(0, loc[1] - guard): loc[1] + guard + 1, :] = -1.0
    second = float(r2.max())

    # Location of the tip inside the new page (pixels -> new-page points)
    tip_px_x = loc[0] + ex0 + (x * old_s - x0)
    tip_px_y = loc[1] + (y * old_s - y0)
    return {"score": float(best), "second": second,
            "tip_new": (tip_px_x / new_s, tip_px_y / new_s)}


def compute_tip_matches(old_doc, new_doc, annots, page_is_image,
                        ANNOT_FREETEXT, ANNOT_SQUARE):
    """Return {annot.index: result} for every annotation on an image page that
    has an anchor point. result keys: page, shift(dx,dy), score, second,
    confident, note."""
    if not HAS_CV:
        return {}

    old_cache, new_cache = {}, {}

    def old_render(pn):
        if pn not in old_cache:
            old_cache[pn] = _render_gray(old_doc[pn])
        return old_cache[pn]

    def new_render(pn):
        if pn not in new_cache:
            new_cache[pn] = _render_gray(new_doc[pn])
        return new_cache[pn]

    out = {}
    for info in annots:
        if not page_is_image.get(info.page_num, False):
            continue
        pt = anchor_for(info, ANNOT_FREETEXT, ANNOT_SQUARE)
        if pt is None or not len(new_doc):
            continue
        old_page = old_doc[info.page_num]
        old_g, old_s = old_render(info.page_num)
        old_pw = old_page.rect.width

        cands = []
        if info.page_num < len(new_doc):
            cands.append(info.page_num)
        cands += [p for p in range(len(new_doc)) if p != info.page_num]

        best = None
        flat = False
        for pn in cands:
            new_g, new_s = new_render(pn)
            m = _match_one(old_g, old_s, new_g, new_s, old_pw, pt)
            if m is None:
                flat = True
                break
            m["page"] = pn
            if best is None or m["score"] > best["score"]:
                best = m
            if m["score"] >= CONFIDENT and (m["score"] - m["second"]) >= MARGIN:
                break

        if best is None:
            out[info.index] = {"page": info.page_num, "shift": None, "score": 0.0,
                               "second": 0.0, "confident": False,
                               "note": "visual: area around the reference is blank — no anchor"}
            continue

        tip_new = best["tip_new"]
        shift = (tip_new[0] - pt[0], tip_new[1] - pt[1])
        confident = (best["score"] >= CONFIDENT and
                     (best["score"] - best["second"]) >= MARGIN)
        note = (f"visual: matched content around the reference "
                f"(score {best['score']:.2f}, shift {shift[1]:+.0f}pt)")
        if not confident:
            if best["score"] < REMOVED_BELOW:
                note = (f"visual: referenced content not found in new PDF "
                        f"(best score {best['score']:.2f})")
            elif best["score"] < CONFIDENT:
                note = f"visual: weak match (score {best['score']:.2f})"
            else:
                note = (f"visual: ambiguous match, similar content repeats "
                        f"(score {best['score']:.2f} vs {best['second']:.2f})")
        out[info.index] = {"page": best["page"], "shift": shift,
                           "score": best["score"], "second": best["second"],
                           "confident": confident, "note": note}

    # Unconfident anchors borrow the shift of the nearest confident neighbour on
    # the same page (content between two anchors moves with them).
    by_page = {}
    for info in annots:
        r = out.get(info.index)
        if r and r["confident"]:
            pt = anchor_for(info, ANNOT_FREETEXT, ANNOT_SQUARE)
            by_page.setdefault(info.page_num, []).append((pt[1], r["shift"], r["page"]))
    for info in annots:
        r = out.get(info.index)
        if not r or r["confident"]:
            continue
        pt = anchor_for(info, ANNOT_FREETEXT, ANNOT_SQUARE)
        neigh = by_page.get(info.page_num)
        if neigh:
            _, sh, pg = min(neigh, key=lambda n: abs(n[0] - pt[1]))
            r["fallback_shift"] = sh
            r["fallback_page"] = pg
        else:
            r["fallback_shift"] = (0.0, 0.0)
            r["fallback_page"] = info.page_num
    return out

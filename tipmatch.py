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


def _render_gray(page, scale=None):
    """Render a page to grayscale WITHOUT annotations. Returns (array, scale)."""
    if scale is None:
        scale = min(1.0, TARGET_PX_W / page.rect.width) if page.rect.width > TARGET_PX_W else 1.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), annots=False,
                          alpha=False, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    fitz.TOOLS.store_shrink(100)
    return arr, scale


def _thumb(page):
    """Small fixed-size grayscale fingerprint of a whole page (annotations off)."""
    sc = 96.0 / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(sc, sc), annots=False, alpha=False,
                          colorspace=fitz.csGRAY)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    fitz.TOOLS.store_shrink(100)      # drop MuPDF's decoded-image cache (keeps RAM flat on big docs)
    a = cv2.resize(a, (48, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    a -= a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 1e-6 else a


def align_pages(old_doc, new_doc, gap=0.25, floor=0.45):
    """Monotonic old->new page mapping (sequence alignment on page thumbnails),
    so inserted or deleted pages don't shift every later page.
    Returns {old_page_index: new_page_index or None}."""
    if not HAS_CV:
        return {}
    no, nn = len(old_doc), len(new_doc)
    if no == 0 or nn == 0:
        return {}
    ot = [_thumb(old_doc[i]) for i in range(no)]
    nt = [_thumb(new_doc[j]) for j in range(nn)]
    S = np.array([[float(np.dot(a, b)) for b in nt] for a in ot])
    NEG = -1e9
    dp = np.zeros((no + 1, nn + 1)); bk = np.zeros((no + 1, nn + 1), dtype=np.int8)
    for i in range(1, no + 1):
        dp[i][0] = dp[i - 1][0] - gap; bk[i][0] = 1
    for j in range(1, nn + 1):
        dp[0][j] = dp[0][j - 1] - gap; bk[0][j] = 2
    for i in range(1, no + 1):
        for j in range(1, nn + 1):
            m = dp[i - 1][j - 1] + (S[i - 1][j - 1] - floor if S[i - 1][j - 1] >= floor else NEG)
            u = dp[i - 1][j] - gap      # old page has no partner (deleted)
            l = dp[i][j - 1] - gap      # new page has no partner (inserted)
            best = max(m, u, l)
            dp[i][j] = best
            bk[i][j] = 0 if best == m else (1 if best == u else 2)
    mapping = {i: None for i in range(no)}
    i, j = no, nn
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bk[i][j] == 0:
            mapping[i - 1] = j - 1; i -= 1; j -= 1
        elif i > 0 and (j == 0 or bk[i][j] == 1):
            i -= 1
        else:
            j -= 1
    return mapping


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
                        ANNOT_FREETEXT, ANNOT_SQUARE, progress=None):
    """Return {annot.index: result} for every annotation on an image page that
    has an anchor point. result keys: page, shift(dx,dy), score, second,
    confident, note."""
    if not HAS_CV:
        return {}

    old_cache, new_cache = {}, {}
    mapping = align_pages(old_doc, new_doc) if (len(old_doc) > 1 or len(new_doc) > 1) else {0: 0}

    def old_render(pn):
        if pn not in old_cache:
            old_cache[pn] = _render_gray(old_doc[pn])
        return old_cache[pn]

    def new_render(pn, old_s, q):
        """New page rendered so content that is q x the old size (in points) shows at the same
        pixel size as the old render. q=1: same artwork, page just re-cut (e.g. narrower);
        q=new_w/old_w: the whole page was scaled."""
        key = (pn, round(old_s, 5), round(q, 4))
        if key not in new_cache:
            sc = min(old_s / q, 3.0)
            new_cache[key] = _render_gray(new_doc[pn], sc)
        return new_cache[key]

    def home_page(old_pn):
        m = mapping.get(old_pn)
        if m is not None:
            return m
        return min(old_pn, len(new_doc) - 1)

    out = {}
    todo = [a for a in annots if page_is_image.get(a.page_num, False)]
    for _n, info in enumerate(todo):
        if progress:
            progress(_n, len(todo))
        if not page_is_image.get(info.page_num, False):
            continue
        pt = anchor_for(info, ANNOT_FREETEXT, ANNOT_SQUARE)
        if pt is None or not len(new_doc):
            continue
        old_page = old_doc[info.page_num]
        old_g, old_s = old_render(info.page_num)
        old_pw = old_page.rect.width

        home = home_page(info.page_num)
        # order: aligned page, then its neighbours, then everything else
        cands = [home] + [p for p in (home - 1, home + 1, home - 2, home + 2)
                          if 0 <= p < len(new_doc)]
        cands += [p for p in range(len(new_doc)) if p not in cands]

        best = None
        flat = False
        for k, pn in enumerate(cands):
            qs = [1.0]
            wr = new_doc[pn].rect.width / old_pw
            if abs(wr - 1.0) > 0.02:
                qs.append(wr)        # page width differs: also try "whole page scaled"
            m = None
            for q in qs:
                new_g, new_s = new_render(pn, old_s, q)
                mq = _match_one(old_g, old_s, new_g, new_s, old_pw, pt)
                if mq is None:
                    m = None
                    break
                if m is None or mq["score"] > m["score"]:
                    m = mq
                if mq["score"] >= 0.9 and (mq["score"] - mq["second"]) >= MARGIN:
                    break
            if m is None:
                flat = True          # patch carries no signal (blank / tiny): same on every page
                break
            m["page"] = pn
            if best is None or m["score"] > best["score"]:
                best = m
            uniq = (m["score"] - m["second"]) >= MARGIN
            # stop early only on a near-identical match; a merely "good" one
            # may be a look-alike on the wrong page, so keep scanning
            if uniq and m["score"] >= 0.95:
                break
            if k == 0 and uniq and m["score"] >= 0.85:
                break                # aligned page matches well: trust it

        if flat or best is None:
            out[info.index] = {"page": home, "shift": None, "score": 0.0, "second": 0.0,
                               "confident": False, "flat": True,
                               "note": "visual: no distinct content at this spot to match — placed relative to the page"}
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
                           "confident": confident, "flat": False, "note": note}

    # Unconfident / flat anchors borrow the shift AND page of the nearest confident
    # neighbour on the same old page (content between anchors moves together).
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
            r["fallback_page"] = min(pg, len(new_doc) - 1)
        else:
            r["fallback_shift"] = (0.0, 0.0)
            r["fallback_page"] = min(home_page(info.page_num), len(new_doc) - 1)
    return out

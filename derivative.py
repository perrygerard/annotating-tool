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


def claim_words(words, tip, boxes=()):
    """words: [(text,x0,y0,x1,y1,linekey)] of the source page, WITHOUT annotation text.
    Returns the words of the OCR lines at the tip (or inside the box the tip is in)."""
    # prefer the highlight box the tip sits inside, if it is a modest size
    inside = [b for b in boxes if b.contains(fitz.Point(*tip)) or _dist(tuple(b), tip) < 4]
    inside.sort(key=lambda b: b.get_area())
    if inside and inside[0].get_area() < 0.35 * 1456 * 1200:
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
        self.generic = {t for t, k in c.items() if k >= 6}

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

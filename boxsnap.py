"""Shrink-wrap a rough rectangle to the content inside it (text or artwork), with even padding.

The user drags a loose box; we render that region of the page WITHOUT annotations, find what differs
from the region's background, and return the tight bounds plus padding. Works on flat (image-only)
pages exactly like on text pages because it looks at pixels, not at a text layer.
"""
import fitz

try:
    import numpy as np
    HAS_NP = True
except Exception:
    HAS_NP = False

MAX_SIDE_PX = 1600      # work resolution for the region
INK_DELTA = 26          # grey-level difference from the background that counts as content
MIN_COUNT = 2           # a row/column needs this many content pixels (ignores specks)


def tighten(pdf_path, page_no, rect, pad=None):
    """rect = fitz.Rect in page points (page_no is 1-based). Returns (fitz.Rect, changed: bool)."""
    if not HAS_NP:
        return fitz.Rect(rect), False
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_no - 1]
        pr = page.rect
        rect = fitz.Rect(rect) & pr
        if rect.is_empty or rect.width < 6 or rect.height < 6:
            return fitz.Rect(rect), False
        scale = min(MAX_SIDE_PX / max(rect.width, rect.height), 4.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, annots=False,
                              alpha=False, colorspace=fitz.csGRAY)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).astype(np.int16)
        fitz.TOOLS.store_shrink(100)
    finally:
        doc.close()

    # Background = the most common grey level in a thin ring around the edge of the region
    ring = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    bg = int(np.bincount((ring // 4).astype(np.int64)).argmax()) * 4 + 2
    ink = np.abs(a - bg) > INK_DELTA
    if ink.mean() > 0.6:                      # photo / busy background: no clear edge to snap to, leave the box as drawn
        return fitz.Rect(rect), False
    cols = np.where(ink.sum(axis=0) >= MIN_COUNT)[0]
    rows = np.where(ink.sum(axis=1) >= MIN_COUNT)[0]
    if len(cols) == 0 or len(rows) == 0:
        return fitz.Rect(rect), False

    x0 = rect.x0 + cols[0] / scale; x1 = rect.x0 + (cols[-1] + 1) / scale
    y0 = rect.y0 + rows[0] / scale; y1 = rect.y0 + (rows[-1] + 1) / scale
    if pad is None:
        pad = max(6.0, pr.width * 0.006)
    tight = fitz.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) & pr
    tight = tight & rect                      # only ever shrink toward the content, never grow past what was drawn
    if tight.width < 8 or tight.height < 8:
        return fitz.Rect(rect), False
    changed = abs(tight.x0 - rect.x0) + abs(tight.y0 - rect.y0) + abs(tight.x1 - rect.x1) + abs(tight.y1 - rect.y1) > 2
    return tight, changed

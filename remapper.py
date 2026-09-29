"""
PDF Annotation Remapper v2
Remaps callout and rectangle annotations from an old PDF to a new version,
preserving metadata (author, dates, reply threads, review status).

Supports both live-text PDFs (text fingerprinting) and image/bitmap PDFs
(visual patch matching via pixel correlation). Detection is automatic.
"""

import fitz  # PyMuPDF
import re
import math
from dataclasses import dataclass, field
from typing import Optional

from tipmatch import compute_tip_matches, HAS_CV as _HAS_TIPMATCH


# Annotation type codes in PyMuPDF
ANNOT_FREETEXT = 2   # Callout
ANNOT_SQUARE   = 4   # Rectangle overlay
ANNOT_POPUP    = 15  # Popup (reply thread carrier)


@dataclass
class AnnotationInfo:
    index: int
    page_num: int
    annot_type: tuple          # (int code, name string)
    rect: fitz.Rect
    content: str
    vertices: list
    flags: int
    colors: dict               # {"stroke": ..., "fill": ...}
    border: dict               # {"width": ..., "style": ...}
    # Metadata to preserve
    author: str = ""
    creation_date: str = ""
    mod_date: str = ""
    subject: str = ""
    annot_id: str = ""
    reply_to: str = ""         # IRT (In-Reply-To) reference
    font_size: float = 7.0     # FreeText font size from DA string
    popup_rect: Optional[fitz.Rect] = None
    open_state: bool = False
    # Fingerprinting
    tip_point: Optional[fitz.Point] = None
    fingerprint_text: str = ""
    fingerprint_rect: Optional[fitz.Rect] = None
    context_before: str = ""   # words before the anchor text
    context_after: str = ""    # words after the anchor text
    # Results
    matched: bool = False
    match_confidence: float = 0.0
    match_method: str = "text"  # "text" | "visual" | "position"
    status: str = "needs_look"  # "moved" | "needs_look" | "content_removed"
    new_rect: Optional[fitz.Rect] = None
    new_vertices: Optional[list] = None
    new_page_num: Optional[int] = None   # page in the new PDF (defaults to page_num)
    match_note: str = ""


def get_words_around_rect(page: fitz.Page, target_rect: fitz.Rect, window: int = 10):
    """Return up to `window` words before and after the text inside target_rect."""
    all_words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,word_idx)
    all_words.sort(key=lambda w: (w[1], w[0]))  # sort by y then x

    # Find words that overlap with the target rect
    target_indices = []
    for i, w in enumerate(all_words):
        wr = fitz.Rect(w[0], w[1], w[2], w[3])
        if wr.intersects(target_rect):
            target_indices.append(i)

    if not target_indices:
        return "", ""

    first_idx = target_indices[0]
    last_idx = target_indices[-1]

    before_words = [all_words[i][4] for i in range(max(0, first_idx - window), first_idx)]
    after_words  = [all_words[i][4] for i in range(last_idx + 1, min(len(all_words), last_idx + window + 1))]

    return " ".join(before_words), " ".join(after_words)


def extract_text_in_rect(page: fitz.Page, rect: fitz.Rect) -> str:
    """Extract all text within a rectangle."""
    words = page.get_text("words", clip=rect)
    return " ".join(w[4] for w in words).strip()


def extract_text_near_point(page: fitz.Page, point: fitz.Point, radius: float = 80):
    search_rect = fitz.Rect(point.x - radius, point.y - radius,
                             point.x + radius, point.y + radius)
    words = page.get_text("words", clip=search_rect)
    if not words and radius < 300:
        return extract_text_near_point(page, point, radius * 2)
    text = " ".join(w[4] for w in words)
    if words:
        actual_rect = fitz.Rect(min(w[0] for w in words), min(w[1] for w in words),
                                max(w[2] for w in words), max(w[3] for w in words))
    else:
        actual_rect = search_rect
    return text.strip(), actual_rect


def clean_fingerprint(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s%(),.-]', '', text)
    return text.strip()


def extract_key_words(text: str, min_len: int = 4) -> list:
    stopwords = {'the', 'and', 'for', 'are', 'was', 'with', 'this', 'that',
                 'from', 'have', 'has', 'been', 'not', 'but', 'also', 'more'}
    words = clean_fingerprint(text).split()
    return [w for w in words if len(w) >= min_len and w not in stopwords]


def text_exists_anywhere(doc: fitz.Document, fingerprint_text: str, key_words: list) -> bool:
    """Return True if any significant portion of fingerprint_text exists in the whole doc."""
    if not key_words:
        return False
    words = fingerprint_text.split()
    for page in doc:
        for window in [4, 3, 2]:
            for i in range(len(words) - window + 1):
                phrase = " ".join(words[i:i+window])
                if len(phrase) >= 6 and page.search_for(phrase):
                    return True
        # Keyword fallback
        hits = sum(1 for kw in key_words[:6] if page.search_for(kw))
        if hits >= min(3, len(key_words)):
            return True
    return False


def find_text_in_page(page: fitz.Page, fingerprint_text: str, key_words: list,
                      context_before: str = "", context_after: str = ""):
    """
    Search for fingerprint_text in page.
    Uses sliding phrase windows, then keyword clustering.
    Context strings boost confidence when they also appear nearby.
    Returns (rect, confidence).
    """
    if not key_words:
        return None, 0.0

    words = fingerprint_text.split()
    best_rect = None
    best_score = 0.0

    # Phrase window search — longest match wins
    for window in [6, 5, 4, 3, 2]:
        for i in range(len(words) - window + 1):
            phrase = " ".join(words[i:i+window])
            if len(phrase) < 8:
                continue
            hits = page.search_for(phrase, quads=False)
            if hits:
                score = window / max(len(words), window)
                # Boost if context words appear nearby
                if context_before or context_after:
                    ctx_words = (context_before + " " + context_after).split()
                    ctx_key = [w for w in ctx_words if len(w) >= 4][:6]
                    nearby_hits = sum(1 for cw in ctx_key if page.search_for(cw))
                    if ctx_key:
                        score += 0.15 * (nearby_hits / len(ctx_key))
                score = min(score, 1.0)
                if score > best_score:
                    best_score = score
                    best_rect = hits[0]
        if best_rect and best_score > 0.5:
            break

    if best_rect:
        return best_rect, best_score

    # Keyword cluster fallback
    hit_rects = []
    for word in key_words[:10]:
        hits = page.search_for(word)
        if hits:
            hit_rects.extend(hits)

    if not hit_rects:
        return None, 0.0

    if len(hit_rects) == 1:
        return hit_rects[0], 0.25

    best_cluster_rect = None
    best_cluster_score = 0.0
    for anchor in hit_rects:
        nearby = [r for r in hit_rects
                  if abs(r.y0 - anchor.y0) < 200 and abs(r.x0 - anchor.x0) < 400]
        if len(nearby) >= 2:
            cluster_rect = fitz.Rect(min(r.x0 for r in nearby), min(r.y0 for r in nearby),
                                     max(r.x1 for r in nearby), max(r.y1 for r in nearby))
            score = len(nearby) / len(key_words)
            if score > best_cluster_score:
                best_cluster_score = score
                best_cluster_rect = cluster_rect

    if best_cluster_rect:
        return best_cluster_rect, min(best_cluster_score, 0.6)

    return hit_rects[0], 0.2


# ── Visual / image-based fingerprinting ─────────────────────────────────────

def is_image_page(page: fitz.Page) -> bool:
    """
    Return True if this page is primarily raster content (a scanned or
    JPG-embedded page) rather than live text.
    We check both: (a) very little extractable text outside annotations, and
    (b) embedded images.

    FreeText annotations render their content as text in the page text stream,
    so we need to exclude annotation text from the count.
    """
    images = page.get_images(full=False)
    if not images:
        return False  # no images — can't be primarily raster
    # Page has images. Check if extractable text is mostly from annotation content.
    full_text = page.get_text().strip()
    # Count chars from all annotation content strings
    annot_chars = sum(
        len(ann.info.get("content", "").strip())
        for ann in page.annots()
    )
    # If nearly all text is annotation-sourced, treat as image page
    if annot_chars > 0 and len(full_text) < annot_chars * 1.4:
        return True
    # Still has significant non-annotation text
    non_annot_chars = max(0, len(full_text) - annot_chars)
    return non_annot_chars <= 60


def render_crop(page: fitz.Page, rect: fitz.Rect, scale: float = 2.0) -> Optional[fitz.Pixmap]:
    """
    Render a cropped region of a page at `scale` times PDF-point resolution.
    Returns None if the rect is empty or out of bounds.
    """
    clipped = rect & page.rect  # intersect with page bounds
    if clipped.is_empty:
        return None
    mat = fitz.Matrix(scale, scale)
    try:
        pix = page.get_pixmap(matrix=mat, clip=clipped, alpha=False)
        return pix
    except Exception:
        return None


def pixmap_to_grayscale_list(pix: fitz.Pixmap) -> list:
    """Convert a pixmap to a flat list of grayscale values (0–255)."""
    # Convert to grayscale in-place if needed
    if pix.n > 1:
        gray = fitz.Pixmap(fitz.csGRAY, pix)
    else:
        gray = pix
    samples = gray.samples  # bytes
    return list(samples)


def downsample_pixels(pixels: list, src_w: int, src_h: int, dst_w: int = 16, dst_h: int = 16) -> list:
    """Bilinear-ish downsampling to a fixed-size thumbnail for comparison."""
    out = []
    for gy in range(dst_h):
        for gx in range(dst_w):
            # Map back to source pixel
            sx = int(gx * src_w / dst_w)
            sy = int(gy * src_h / dst_h)
            # Clamp
            sx = min(sx, src_w - 1)
            sy = min(sy, src_h - 1)
            out.append(pixels[sy * src_w + sx])
    return out


def pixel_hash(pix: fitz.Pixmap, size: int = 16) -> list:
    """
    Compute a simple perceptual hash: downsample to size×size grayscale,
    return list of 0/1 values (above/below median).
    """
    gray_pixels = pixmap_to_grayscale_list(pix)
    thumb = downsample_pixels(gray_pixels, pix.width, pix.height, size, size)
    median = sorted(thumb)[len(thumb) // 2]
    return [1 if v >= median else 0 for v in thumb]


def hamming_similarity(h1: list, h2: list) -> float:
    """Return similarity 0–1 based on Hamming distance between two hashes."""
    if len(h1) != len(h2) or not h1:
        return 0.0
    matches = sum(a == b for a, b in zip(h1, h2))
    return matches / len(h1)


def ncc_similarity(pix1: fitz.Pixmap, pix2: fitz.Pixmap) -> float:
    """
    Normalized cross-correlation between two pixmaps after downsampling to 16×16.
    Returns value 0–1 (1 = identical).
    """
    if pix1 is None or pix2 is None:
        return 0.0
    g1 = pixmap_to_grayscale_list(pix1)
    g2 = pixmap_to_grayscale_list(pix2)
    # Downsample both to 16×16
    t1 = downsample_pixels(g1, pix1.width, pix1.height, 16, 16)
    t2 = downsample_pixels(g2, pix2.width, pix2.height, 16, 16)
    n = len(t1)
    mean1 = sum(t1) / n
    mean2 = sum(t2) / n
    num = sum((a - mean1) * (b - mean2) for a, b in zip(t1, t2))
    den1 = math.sqrt(sum((a - mean1) ** 2 for a in t1))
    den2 = math.sqrt(sum((b - mean2) ** 2 for b in t2))
    if den1 < 1e-6 or den2 < 1e-6:
        return 0.0
    ncc = num / (den1 * den2)
    return (ncc + 1.0) / 2.0  # map [-1,1] → [0,1]


def detect_removal_boundary(old_page: fitz.Page, new_page: fitz.Page,
                            scale: float = 0.1) -> tuple:
    """
    Find where content was removed between old and new page, and the shift amount.

    Strategy: render both pages without annotations, compare column-by-column
    running similarity, find the y-transition where old rows stop matching
    new rows at the same position and start matching offset rows.

    Returns (boundary_y, shift_y) in PDF points:
      - boundary_y: approximate y in old PDF where the content shift begins
      - shift_y: how much content below boundary_y has shifted (negative = upward)
    Returns (None, 0) if no significant shift is detected.
    """
    mat = fitz.Matrix(scale, scale)
    try:
        old_pix = old_page.get_pixmap(matrix=mat, alpha=False, annots=False)
        new_pix = new_page.get_pixmap(matrix=mat, alpha=False, annots=False)
    except Exception:
        return (None, 0)

    oh, ow = old_pix.height, old_pix.width
    nh, nw = new_pix.height, new_pix.width

    old_bytes = bytearray(old_pix.samples)
    new_bytes = bytearray(new_pix.samples)
    n_ch = old_pix.n

    def row_mean(data, row, w, ch):
        s = row * w * ch
        vals = data[s:s + w * ch]
        return sum(vals) / len(vals) if vals else 128.0

    # Build row fingerprints for both pages
    old_rows = [row_mean(old_bytes, r, ow, n_ch) for r in range(oh)]
    new_rows = [row_mean(new_bytes, r, nw, n_ch) for r in range(nh)]

    # For each old row, find what new row it matches best (within search window)
    SEARCH_WINDOW = int(oh * 0.7)

    # Find the shift at the bottom half of the page — that gives us the net removed height
    bottom_old = range(max(0, oh * 2 // 3), oh, max(1, oh // 20))
    shift_votes_bottom = []
    for old_r in bottom_old:
        best_r = old_r
        best_d = float('inf')
        for new_r in range(max(0, old_r - SEARCH_WINDOW), min(nh, old_r + 10)):
            d = abs(old_rows[old_r] - new_rows[new_r])
            if d < best_d:
                best_d = d
                best_r = new_r
        if best_d < 8:
            shift_votes_bottom.append(best_r - old_r)

    # Find the shift at the top third of the page — should be ~0 if removal is in middle
    top_old = range(0, oh // 3, max(1, oh // 20))
    shift_votes_top = []
    for old_r in top_old:
        best_r = old_r
        best_d = float('inf')
        for new_r in range(max(0, old_r - 10), min(nh, old_r + 10)):
            d = abs(old_rows[old_r] - new_rows[new_r])
            if d < best_d:
                best_d = d
                best_r = new_r
        if best_d < 8:
            shift_votes_top.append(best_r - old_r)

    if not shift_votes_bottom:
        return (None, 0)

    def median(lst):
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n else 0

    top_shift = median(shift_votes_top) if shift_votes_top else 0
    bottom_shift = median(shift_votes_bottom)
    net_shift_px = bottom_shift - top_shift

    if abs(net_shift_px) < 3:
        return (None, 0)  # No significant shift

    # Find the transition boundary: scan down from top until the shift changes
    boundary_old_r = oh // 2  # default: middle of page
    prev_shift = top_shift
    for old_r in range(0, oh, max(1, oh // 50)):
        search_lo = max(0, old_r + prev_shift - 5)
        search_hi = min(nh - 1, old_r + bottom_shift + 5)
        best_r = old_r
        best_d = float('inf')
        for new_r in range(search_lo, search_hi + 1):
            d = abs(old_rows[old_r] - new_rows[new_r])
            if d < best_d:
                best_d = d
                best_r = new_r
        if best_d < 8:
            cur_shift = best_r - old_r
            if abs(cur_shift - top_shift) >= abs(net_shift_px) * 0.5:
                boundary_old_r = old_r
                break

    boundary_y = boundary_old_r / scale
    shift_y = net_shift_px / scale
    return (boundary_y, shift_y)


def detect_vertical_shifts(old_page: fitz.Page, new_page: fitz.Page,
                           scale: float = 0.15) -> list:
    """
    Compare old and new page row-by-row to detect where content was inserted
    or removed, and by how much.

    Returns a list of (y_threshold, y_shift) tuples, sorted by y_threshold.
    For a given annotation at old_y:
      - Find the last entry where y_threshold <= old_y
      - Apply that entry's y_shift to get new_y
    y_shift is negative when content above was removed (annotations shift up).

    Uses low-resolution rendering for speed.
    """
    mat = fitz.Matrix(scale, scale)
    try:
        old_pix = old_page.get_pixmap(matrix=mat, alpha=False)
        new_pix = new_page.get_pixmap(matrix=mat, alpha=False)
    except Exception:
        return [(0, 0)]

    oh, ow = old_pix.height, old_pix.width
    nh, nw = new_pix.height, new_pix.width

    # Convert to flat byte arrays
    old_bytes = old_pix.samples
    new_bytes = new_pix.samples
    n_channels = old_pix.n

    def row_avg(data, row, w, ch):
        start = row * w * ch
        end = start + w * ch
        return sum(data[start:end]) / (w * ch) if w * ch > 0 else 128.0

    # For each row in new, find the best matching row in old
    # Build a mapping: new_row -> best_old_row
    # Then derive: for old_row R, what new_row does it map to?

    SEARCH_RADIUS = int(oh * 0.6)  # search up to 60% of page height away

    shifts = [(0, 0)]  # (y_threshold_pdf, shift_pdf) — start with no shift

    # Scan old rows in steps, find where shift starts changing
    prev_shift_px = 0
    shift_start_old_row = None

    for old_r in range(0, oh, max(1, oh // 80)):
        old_val = row_avg(old_bytes, old_r, ow, n_channels)
        best_new_r = old_r
        best_diff = float('inf')

        search_lo = max(0, old_r - SEARCH_RADIUS)
        search_hi = min(nh - 1, old_r + SEARCH_RADIUS)
        # Bias search toward current known shift
        center = old_r + prev_shift_px
        ordered = sorted(range(search_lo, search_hi + 1),
                         key=lambda r: abs(r - center))

        for new_r in ordered[:60]:  # check up to 60 candidates
            new_val = row_avg(new_bytes, new_r, nw, n_channels)
            diff = abs(old_val - new_val)
            if diff < best_diff:
                best_diff = diff
                best_new_r = new_r
            if diff < 1.0:
                break

        if best_diff > 20:
            continue  # row too different to use as reference

        cur_shift_px = best_new_r - old_r

        if abs(cur_shift_px - prev_shift_px) >= 3:
            # Shift changed — record the transition
            y_threshold_pdf = old_r / scale
            y_shift_pdf = cur_shift_px / scale
            # Only add if meaningfully different from last recorded shift
            if not shifts or abs(shifts[-1][1] - y_shift_pdf) >= 10:
                shifts.append((y_threshold_pdf, y_shift_pdf))
            prev_shift_px = cur_shift_px

    return sorted(shifts, key=lambda x: x[0])


def get_shift_for_y(shifts: list, old_y: float) -> float:
    """
    Given the shift table from detect_vertical_shifts and a y-coordinate in
    the old PDF, return the y shift to apply.
    """
    result = 0.0
    for y_thresh, y_shift in shifts:
        if old_y >= y_thresh:
            result = y_shift
        else:
            break
    return result


def find_visual_match(old_page: fitz.Page, new_doc: fitz.Document,
                      annot_rect: fitz.Rect,
                      start_page: int = 0,
                      shift_table: list = None) -> tuple:
    """
    Try to locate the visual region around `annot_rect` on old_page
    somewhere in new_doc using pixel correlation.

    Returns (page_num, new_rect, confidence, note).
    - new_rect is placed at the same relative position (normalized coords)
      unless a better match is found by sliding search.
    - confidence: 0.0–1.0
    """
    old_pw, old_ph = old_page.rect.width, old_page.rect.height
    annot_w = annot_rect.x1 - annot_rect.x0
    annot_h = annot_rect.y1 - annot_rect.y0
    # Normalized annotation position in old page
    norm_x = annot_rect.x0 / old_pw
    norm_y = annot_rect.y0 / old_ph
    norm_w = annot_w / old_pw
    norm_h = annot_h / old_ph

    # ── Build fingerprint from SLIDE BODY CONTENT near the annotation ─────────
    # Annotations sit in margins; slide body content is in the center.
    # Sample a wide center strip at the same vertical position.
    CONTEXT_PAD = 30  # PDF points
    # Use the inner 60% of the page width as the "slide body" reference strip
    body_x0 = old_pw * 0.15
    body_x1 = old_pw * 0.85
    if annot_rect.x0 < old_pw * 0.3:
        # Left-margin annotation: sample body to the right
        content_x0 = max(annot_rect.x1 + 30, body_x0)
        content_x1 = body_x1
    elif annot_rect.x1 > old_pw * 0.7:
        # Right-margin annotation: sample body to the left
        content_x0 = body_x0
        content_x1 = min(annot_rect.x0 - 30, body_x1)
    else:
        # Center annotation: use wider body strip
        content_x0 = body_x0
        content_x1 = body_x1

    content_crop_rect = fitz.Rect(
        content_x0,
        annot_rect.y0 - CONTEXT_PAD,
        content_x1,
        annot_rect.y1 + CONTEXT_PAD,
    )
    # Render WITHOUT annotations so the patch matches the clean new PDF
    old_crop = None
    try:
        clipped = content_crop_rect & old_page.rect
        if not clipped.is_empty:
            mat = fitz.Matrix(1.5, 1.5)
            old_crop = old_page.get_pixmap(matrix=mat, clip=clipped, alpha=False, annots=False)
    except Exception:
        pass

    if old_crop is None:
        # Fall back to annotation area
        crop_rect = fitz.Rect(
            annot_rect.x0 - CONTEXT_PAD,
            annot_rect.y0 - CONTEXT_PAD,
            annot_rect.x1 + CONTEXT_PAD,
            annot_rect.y1 + CONTEXT_PAD,
        )
        old_crop = render_crop(old_page, crop_rect, scale=1.5)
    if old_crop is None:
        return None, None, 0.0, "visual: crop failed"

    old_hash = pixel_hash(old_crop)

    best_page = None
    best_rect = None
    best_conf = 0.0
    best_note = ""

    # Search each candidate page
    candidate_pages = list(range(start_page, len(new_doc)))
    # Put same-index page first
    if start_page < len(new_doc):
        candidate_pages = [start_page] + [p for p in range(len(new_doc)) if p != start_page]

    for new_page_num in candidate_pages:
        new_page = new_doc[new_page_num]
        npw, nph = new_page.rect.width, new_page.rect.height

        # ── Primary strategy: vertical strip search using adjacent content ──────
        # Slide the content crop vertically through the new page to find where
        # the annotation's adjacent slide content best matches.
        # This works for both image and non-image pages when a shift_table hint
        # is available or not.
        strip_h = max(annot_h + 2 * CONTEXT_PAD, 80)
        strip_rect_old = fitz.Rect(content_x0, annot_rect.y0 - CONTEXT_PAD,
                                   content_x1, annot_rect.y0 - CONTEXT_PAD + strip_h)
        old_content_crop = None
        try:
            clipped = strip_rect_old & old_page.rect
            if not clipped.is_empty:
                old_content_crop = old_page.get_pixmap(
                    matrix=fitz.Matrix(1.5, 1.5), clip=clipped, alpha=False, annots=False)
        except Exception:
            pass

        if old_content_crop is not None:
            old_content_hash = pixel_hash(old_content_crop, size=12)
            # Search vertically: use shift_table hint to narrow range
            if shift_table is not None:
                hint_shift = get_shift_for_y(shift_table, annot_rect.y0)
                search_lo = max(0, annot_rect.y0 + hint_shift - 400)
                search_hi = min(nph - strip_h, annot_rect.y0 + hint_shift + 400)
            else:
                search_lo = 0
                search_hi = nph - strip_h

            VSTEP = max(strip_h * 0.15, 20)  # step 15% of strip height
            v_best_conf = 0.0
            v_best_y = annot_rect.y0  # fallback: same position
            v_best_y += (hint_shift if shift_table else 0)

            y = search_lo
            while y <= search_hi:
                strip_rect_new = fitz.Rect(content_x0, y, content_x1, y + strip_h)
                new_content_crop = None
                try:
                    clipped2 = strip_rect_new & new_page.rect
                    if not clipped2.is_empty:
                        new_content_crop = new_page.get_pixmap(
                            matrix=fitz.Matrix(1.5, 1.5), clip=clipped2, alpha=False, annots=False)
                except Exception:
                    pass
                if new_content_crop is not None:
                    new_content_hash = pixel_hash(new_content_crop, size=12)
                    sim = hamming_similarity(old_content_hash, new_content_hash)
                    if sim > v_best_conf:
                        v_best_conf = sim
                        v_best_y = y + CONTEXT_PAD  # remove pad offset → new annot y0
                y += VSTEP

            y_shift = v_best_y - annot_rect.y0
            shifted_y0 = max(0, min(v_best_y, nph - annot_h))
            shifted_y1 = shifted_y0 + annot_h
            # Use strip similarity directly as confidence
            conf = v_best_conf
            shifted_rect = fitz.Rect(annot_rect.x0, shifted_y0, annot_rect.x1, shifted_y1)
            if conf > best_conf:
                best_conf = conf
                best_page = new_page_num
                best_rect = shifted_rect
                best_note = (f"visual: strip-search y_shift={y_shift:+.0f} "
                             f"page {new_page_num + 1} (conf={conf:.0%})")
            if conf >= 0.70:
                break
            continue

        # ── Fallback: sliding search (no shift table) ─────────────────────────
        # First: try the same normalized position (fast path)
        candidate_rect = fitz.Rect(
            norm_x * npw - CONTEXT_PAD,
            norm_y * nph - CONTEXT_PAD,
            (norm_x + norm_w) * npw + CONTEXT_PAD,
            (norm_y + norm_h) * nph + CONTEXT_PAD,
        )
        new_crop = render_crop(new_page, candidate_rect, scale=1.5)
        same_pos_conf = 0.0
        if new_crop is not None:
            new_hash = pixel_hash(new_crop)
            same_pos_conf = hamming_similarity(old_hash, new_hash)

        if same_pos_conf >= 0.82:
            # Strong match at same position — accept immediately
            annot_new_rect = fitz.Rect(
                norm_x * npw, norm_y * nph,
                (norm_x + norm_w) * npw, (norm_y + norm_h) * nph,
            )
            if same_pos_conf > best_conf:
                best_conf = same_pos_conf
                best_page = new_page_num
                best_rect = annot_new_rect
                best_note = f"visual: same-position match page {new_page_num + 1} ({same_pos_conf:.0%})"
            break  # don't bother sliding

        # Sliding search — scan a grid of positions on the new page
        # Use coarse step to keep it fast
        STEP = max(annot_w * 0.5, 40)  # slide in 50%-width steps
        SEARCH_W = npw
        SEARCH_H = nph

        slide_best_conf = same_pos_conf
        slide_best_rect = fitz.Rect(
            norm_x * npw, norm_y * nph,
            (norm_x + norm_w) * npw, (norm_y + norm_h) * nph,
        )

        x = 0.0
        while x + annot_w <= SEARCH_W:
            y = 0.0
            while y + annot_h <= SEARCH_H:
                test_crop_rect = fitz.Rect(
                    x - CONTEXT_PAD, y - CONTEXT_PAD,
                    x + annot_w + CONTEXT_PAD, y + annot_h + CONTEXT_PAD
                )
                test_crop = render_crop(new_page, test_crop_rect, scale=1.0)
                if test_crop is not None:
                    test_hash = pixel_hash(test_crop)
                    conf = hamming_similarity(old_hash, test_hash)
                    if conf > slide_best_conf:
                        slide_best_conf = conf
                        slide_best_rect = fitz.Rect(x, y, x + annot_w, y + annot_h)
                y += STEP
            x += STEP

        if slide_best_conf > best_conf:
            best_conf = slide_best_conf
            best_page = new_page_num
            best_rect = slide_best_rect
            if slide_best_conf >= 0.72:
                best_note = f"visual: slide match page {new_page_num + 1} ({slide_best_conf:.0%})"
            else:
                best_note = f"visual: weak match page {new_page_num + 1} ({slide_best_conf:.0%})"

        # Once we've checked the same-index page and it's decent, stop
        if best_conf >= 0.65 and new_page_num == start_page:
            break

    if best_page is None or best_conf < 0.45:
        # Fall back: keep original normalized position
        new_page = new_doc[start_page] if start_page < len(new_doc) else new_doc[0]
        npw, nph = new_page.rect.width, new_page.rect.height
        fallback_rect = fitz.Rect(
            norm_x * npw, norm_y * nph,
            (norm_x + norm_w) * npw, (norm_y + norm_h) * nph,
        )
        return start_page, fallback_rect, best_conf, "visual: low confidence — kept original position"

    return best_page, best_rect, best_conf, best_note


def apply_offset(rect: fitz.Rect, offset: fitz.Point) -> fitz.Rect:
    return fitz.Rect(rect.x0 + offset.x, rect.y0 + offset.y,
                     rect.x1 + offset.x, rect.y1 + offset.y)


def copy_annot_metadata(src_info: dict, annot: fitz.Annot):
    """Write preserved metadata back onto a newly created annotation."""
    info = annot.info
    if src_info.get("author"):
        info["title"] = src_info["author"]
    if src_info.get("content"):
        info["content"] = src_info["content"]
    if src_info.get("subject"):
        info["subject"] = src_info["subject"]
    if src_info.get("creation_date"):
        info["creationDate"] = src_info["creation_date"]
    if src_info.get("mod_date"):
        info["modDate"] = src_info["mod_date"]
    annot.set_info(info)


def process_pdfs(old_pdf_path: str, new_pdf_path: str, output_path: str) -> dict:
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)

    results = {
        "total": 0,
        "matched": 0,
        "unmatched": 0,
        "moved": 0,
        "needs_look": 0,
        "content_removed": 0,
        "skipped": [],
        "annotations": []
    }

    # ── 1. Extract all annotations from old doc ──────────────────────────────
    all_annots = []
    next_index = 0
    for page_num in range(len(old_doc)):
        page = old_doc[page_num]
        for i, annot in enumerate(page.annots()):
            atype = annot.type  # (int, name)
            # Skip popups — they travel with their parent annotation
            if atype[0] == ANNOT_POPUP:
                continue
            # Only text callouts and rectangles are remapped. Anything else (bare arrow
            # lines, highlights, ...) is reported back instead of being silently dropped.
            if atype[0] not in (ANNOT_FREETEXT, ANNOT_SQUARE):
                results["skipped"].append({"page": page_num + 1, "type": atype[1]})
                continue

            vertices = list(annot.vertices or [])
            if not vertices and atype[0] == ANNOT_FREETEXT:
                # Callout tip lives in /CL (PDF coords, y-up) when /Vertices is absent
                try:
                    cl = old_doc.xref_get_key(annot.xref, "CL")[1]
                    nums = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', cl or "")]
                    if len(nums) >= 4:
                        ph_old = page.mediabox.height
                        vertices = [(nums[k] - page.mediabox.x0,
                                     ph_old - (nums[k + 1] - page.mediabox.y0))
                                    for k in range(0, len(nums) - 1, 2)]
                except Exception:
                    pass
            tip_point = fitz.Point(vertices[0][0], vertices[0][1]) if vertices else None

            annot_info_dict = annot.info
            colors = annot.colors  # {"stroke": rgb_tuple, "fill": rgb_tuple}
            border = annot.border  # {"width": float, "style": str}

            # Extract font size from the DA (Default Appearance) string, e.g. "/Helv 8 Tf"
            font_size = 7.0
            try:
                da = annot.get_oc()  # may not exist
            except Exception:
                da = None
            try:
                # DA string is accessible via the annotation's xref in the PDF
                xref = annot.xref
                da_str = old_doc.xref_get_key(xref, "DA")[1] if xref else ""
                if da_str:
                    import re as _re
                    m = _re.search(r'(\d+(?:\.\d+)?)\s+Tf', da_str)
                    if m:
                        font_size = float(m.group(1))
            except Exception:
                pass

            info = AnnotationInfo(
                index=next_index,
                page_num=page_num,
                annot_type=atype,
                rect=fitz.Rect(annot.rect),
                content=annot_info_dict.get("content", ""),
                vertices=vertices,
                flags=annot.flags,
                colors=colors,
                border=border,
                author=annot_info_dict.get("title", ""),
                creation_date=annot_info_dict.get("creationDate", ""),
                mod_date=annot_info_dict.get("modDate", ""),
                subject=annot_info_dict.get("subject", ""),
                annot_id=annot_info_dict.get("id", ""),
                reply_to=annot_info_dict.get("irt", ""),
                open_state=annot.is_open,
                tip_point=tip_point,
                font_size=font_size,
            )

            # ── Fingerprint strategy ──────────────────────────────────────────
            # Priority: text AT the arrow tip / inside the rect (what the annotation
            # is POINTING TO) — this moves correctly even when sections are removed
            # and content shifts. The annotation content string (REF label) is used
            # only as supplementary context, not as the primary anchor.

            # For rectangles: use text INSIDE the rect as the anchor
            if atype[0] == ANNOT_SQUARE:
                rect_text = extract_text_in_rect(page, info.rect)
                if rect_text and len(rect_text) > 4:
                    info.fingerprint_text = rect_text
                    info.fingerprint_rect = fitz.Rect(info.rect)
                    ctx_b, ctx_a = get_words_around_rect(page, info.rect)
                    info.context_before = ctx_b
                    info.context_after = ctx_a

            # For callouts: PRIMARY anchor = text at the arrow tip point.
            # The annotation content text (e.g. REF strings) is stored separately
            # and used only as context to boost match confidence.
            if not info.fingerprint_text:
                # 1. Try tip point first (where the arrow is pointing)
                if tip_point:
                    fp_text, fp_rect = extract_text_near_point(page, tip_point)
                    if fp_text and len(fp_text) > 4:
                        info.fingerprint_text = fp_text
                        info.fingerprint_rect = fp_rect
                        if fp_rect:
                            ctx_b, ctx_a = get_words_around_rect(page, fp_rect)
                            info.context_before = ctx_b
                            info.context_after = ctx_a

                # 2. Try the last vertex (anchor end of callout line) if tip gave nothing
                if not info.fingerprint_text and vertices and len(vertices) >= 2:
                    anchor = fitz.Point(vertices[-1][0], vertices[-1][1])
                    fp_text, fp_rect = extract_text_near_point(page, anchor)
                    if fp_text and len(fp_text) > 4:
                        info.fingerprint_text = fp_text
                        info.fingerprint_rect = fp_rect
                        if fp_rect:
                            ctx_b, ctx_a = get_words_around_rect(page, fp_rect)
                            info.context_before = ctx_b
                            info.context_after = ctx_a

                # 3. Last resort: fall back to annotation content text (REF label etc.)
                #    This is least reliable for repositioning but better than nothing.
                if not info.fingerprint_text:
                    content_text = info.content.strip()
                    if content_text and len(content_text) > 8:
                        info.fingerprint_text = content_text
                        info.fingerprint_rect = None

            # Store annotation content as extra context to disambiguate same-text matches
            # (e.g. the same phrase appears twice but only one has this REF nearby)
            if info.content.strip() and info.content.strip() != info.fingerprint_text:
                content_words = info.content.strip().split()
                extra_ctx = " ".join(content_words[:20])  # first 20 words of REF label
                if info.context_after:
                    info.context_after = info.context_after + " " + extra_ctx
                else:
                    info.context_after = extra_ctx

            next_index += 1
            all_annots.append(info)

    results["total"] = len(all_annots)

    # ── 2. Detect content type per page (text vs image) ─────────────────────
    page_is_image = {}
    for page_num in range(len(old_doc)):
        page_is_image[page_num] = is_image_page(old_doc[page_num])

    # ── 2b. Legacy page-wide removal boundary (only computed if a page needs it) ─
    page_removal_info = {}

    def get_removal_info(old_page_num):
        if old_page_num not in page_removal_info:
            old_page = old_doc[old_page_num]
            new_page_num = old_page_num if old_page_num < len(new_doc) else len(new_doc) - 1
            page_removal_info[old_page_num] = detect_removal_boundary(old_page, new_doc[new_page_num])
        return page_removal_info[old_page_num]

    # ── 2c. Tip-anchored visual matching (image pages) ───────────────────────
    # Match the pixels around each annotation's referenced point, per annotation.
    tip_matches = compute_tip_matches(old_doc, new_doc, all_annots, page_is_image,
                                      ANNOT_FREETEXT, ANNOT_SQUARE)

    # ── 3. Match each annotation in the new doc ───────────────────────────────
    for info in all_annots:
        key_words = extract_key_words(info.fingerprint_text) if info.fingerprint_text else []
        page_image_mode = page_is_image.get(info.page_num, False)

        # Check if the content still exists anywhere in the new doc (text mode only)
        if info.fingerprint_text and key_words and not page_image_mode:
            still_exists = text_exists_anywhere(new_doc, info.fingerprint_text, key_words)
        else:
            still_exists = True  # can't determine for image pages

        tm = tip_matches.get(info.index)
        if tm is not None:
            # ── Tip-anchored visual match (preferred for bitmap pages) ───────
            if tm["confident"]:
                dx, dy = tm["shift"]
                target_page = tm["page"]
                info.matched = True
                info.match_confidence = tm["score"]
                info.status = "moved" if tm["score"] >= 0.85 else "needs_look"
            else:
                # Weak / ambiguous / not found: borrow the nearest confident
                # neighbour's shift so the dot still lands in the right region.
                dx, dy = tm["fallback_shift"]
                target_page = tm["fallback_page"]
                info.matched = False
                info.match_confidence = tm["score"]
                if tm["score"] < 0.5 and not tm.get("flat"):
                    info.status = "content_removed"
                    results["content_removed"] += 1
                else:
                    info.status = "needs_look"
            info.match_method = "visual"
            info.match_note = tm["note"]
            info.new_rect = apply_offset(info.rect, fitz.Point(dx, dy))
            info.new_vertices = [(v[0] + dx, v[1] + dy) for v in info.vertices]
            info.new_page_num = target_page

        # ── Image-mode: use removal boundary + visual validation ─────────────
        elif page_image_mode or (not info.fingerprint_text and not key_words):
            if page_image_mode:
                old_page = old_doc[info.page_num]
                new_pn = info.page_num if info.page_num < len(new_doc) else len(new_doc) - 1
                removal_info = get_removal_info(info.page_num)
                boundary_y, shift_y = removal_info

                # Determine y-shift for this annotation
                annot_center_y = (info.rect.y0 + info.rect.y1) / 2
                if boundary_y is not None and annot_center_y > boundary_y:
                    # Annotation is below the removed region — shift it up
                    y_offset = shift_y
                    confidence = 0.80
                    note = (f"visual: removed-block boundary y={boundary_y:.0f}, "
                            f"shift={shift_y:+.0f}")
                elif boundary_y is not None:
                    # Annotation is above the removed region — no shift
                    y_offset = 0
                    confidence = 0.85
                    note = f"visual: above removal boundary (y={boundary_y:.0f})"
                else:
                    # No removal detected — try visual patch matching
                    new_pn2, new_rect2, confidence, note = find_visual_match(
                        old_page, new_doc, info.rect, start_page=info.page_num,
                        shift_table=None)
                    if new_rect2 is not None:
                        info.new_rect = new_rect2
                        info.new_vertices = [
                            (v[0] + (new_rect2.x0 - info.rect.x0),
                             v[1] + (new_rect2.y0 - info.rect.y0))
                            for v in info.vertices
                        ]
                        info.matched = confidence >= 0.55
                        info.match_confidence = confidence
                        info.match_method = "visual"
                        info.match_note = note
                        info.status = "moved" if confidence >= 0.72 else "needs_look"
                    y_offset = 0  # handled above

                if boundary_y is not None:
                    # Apply the computed y-offset
                    new_doc_page = new_doc[new_pn]
                    nph = new_doc_page.rect.height
                    annot_h = info.rect.y1 - info.rect.y0
                    new_y0 = max(0, min(info.rect.y0 + y_offset, nph - annot_h))
                    new_y1 = new_y0 + annot_h
                    new_rect = fitz.Rect(info.rect.x0, new_y0, info.rect.x1, new_y1)
                    info.new_rect = new_rect
                    info.new_vertices = [
                        (v[0], v[1] + y_offset)
                        for v in info.vertices
                    ]
                    info.matched = True
                    info.match_confidence = confidence
                    info.match_method = "visual"
                    info.match_note = note
                    # Annotations whose center was inside the removed zone
                    # (between boundary and boundary - shift) need review
                    removed_zone_top = boundary_y + shift_y  # top of removed block in old PDF
                    removed_zone_bot = boundary_y
                    if (removed_zone_top <= annot_center_y <= removed_zone_bot or
                            removed_zone_bot <= annot_center_y <= removed_zone_top):
                        info.status = "needs_look"
                        info.match_note += " — annotation may have pointed at removed content"
                    else:
                        info.status = "moved"
            else:
                # No fingerprint and not image — keep original position
                if info.page_num < len(new_doc):
                    info.matched = True
                    info.match_confidence = 0.5
                    info.match_method = "position"
                    info.new_rect = fitz.Rect(info.rect)
                    info.new_vertices = list(info.vertices)
                    info.status = "needs_look"
                    info.match_note = "No text anchor — kept original position"

        else:
            # ── Text-mode: try same page first, then other pages ─────────────
            target_pages = []
            if info.page_num < len(new_doc):
                target_pages.append(info.page_num)
            target_pages += [p for p in range(len(new_doc)) if p != info.page_num]

            for new_page_num in target_pages:
                new_page = new_doc[new_page_num]
                match_rect, confidence = find_text_in_page(
                    new_page, info.fingerprint_text, key_words,
                    info.context_before, info.context_after)

                if match_rect and confidence > 0.15:
                    # Compute offset from old anchor → new anchor
                    if info.fingerprint_rect:
                        old_center = fitz.Point(
                            (info.fingerprint_rect.x0 + info.fingerprint_rect.x1) / 2,
                            (info.fingerprint_rect.y0 + info.fingerprint_rect.y1) / 2)
                    else:
                        old_page = old_doc[info.page_num]
                        old_match, _ = find_text_in_page(
                            old_page, info.fingerprint_text, key_words)
                        old_center = fitz.Point(
                            (old_match.x0 + old_match.x1) / 2,
                            (old_match.y0 + old_match.y1) / 2) if old_match else fitz.Point(
                            (info.rect.x0 + info.rect.x1) / 2,
                            (info.rect.y0 + info.rect.y1) / 2)

                    new_center = fitz.Point(
                        (match_rect.x0 + match_rect.x1) / 2,
                        (match_rect.y0 + match_rect.y1) / 2)
                    offset = fitz.Point(new_center.x - old_center.x,
                                        new_center.y - old_center.y)

                    info.new_rect = apply_offset(info.rect, offset)
                    info.new_vertices = [(v[0] + offset.x, v[1] + offset.y)
                                         for v in info.vertices]
                    info.matched = True
                    info.match_confidence = confidence
                    info.match_method = "text"
                    info.status = "moved"
                    info.match_note = (f"Moved to page {new_page_num + 1} "
                                       f"(confidence {confidence:.0%})")
                    break

        # ── Classify unmatched ────────────────────────────────────────────────
        if not info.matched and info.new_rect is None:
            if not still_exists and not page_image_mode:
                info.status = "content_removed"
                info.match_note = "Annotated content no longer exists in new PDF"
                results["content_removed"] += 1
            else:
                info.status = "needs_look"
                info.match_note = "Position could not be determined — kept original"
            # Keep at original position so nothing is silently lost
            if info.page_num < len(new_doc):
                info.new_rect = fitz.Rect(info.rect)
                info.new_vertices = list(info.vertices)

        # Low-confidence matches also need a look
        if info.matched and info.match_confidence < 0.4:
            info.status = "needs_look"

        # Tally status counters
        if info.status == "moved":
            results["moved"] += 1
        elif info.status == "needs_look":
            results["needs_look"] += 1
        elif info.status == "content_removed":
            pass  # already incremented above when classified

        results["matched" if info.matched else "unmatched"] += 1
        results["annotations"].append({
            "index": info.index,
            "page": info.page_num + 1,
            "type": info.annot_type[1],
            "content": info.content[:80] + ("..." if len(info.content) > 80 else ""),
            "author": info.author,
            "status": info.status,
            "confidence": f"{info.match_confidence:.0%}",
            "method": info.match_method,
            "note": info.match_note,
        })

    num_by_index = assign_numbers(all_annots)
    for info, entry in zip(all_annots, results["annotations"]):
        entry["number"] = num_by_index.get(info.index)

    # ── 4. Build page preview data (dimensions + annotation overlays) ─────────
    # Group annotations by page for the preview renderer
    annots_by_page = {}
    for info in all_annots:
        key = (info.page_num, _out_page(info))     # (page in old PDF, page in new PDF)
        annots_by_page.setdefault(key, []).append(info)

    pages_preview = []
    preview_doc = fitz.open(new_pdf_path)
    old_preview_doc = fitz.open(old_pdf_path)
    for (page_num, out_pn) in sorted(annots_by_page.keys()):
        if out_pn >= len(preview_doc):
            continue
        page = preview_doc[out_pn]
        pw, ph = page.rect.width, page.rect.height
        S_pg = _scale(pw)
        tot_w = pw + SIDEBAR_BASE * S_pg      # output page = original + sidebar
        dot_r_pg = DOT_BASE * S_pg

        # Old page dimensions (may differ from new page)
        old_pw, old_ph = pw, ph
        if page_num < len(old_preview_doc):
            old_page = old_preview_doc[page_num]
            old_pw, old_ph = old_page.rect.width, old_page.rect.height

        overlay_annots = []
        for info in annots_by_page[(page_num, out_pn)]:
            new_r = info.new_rect if info.new_rect else info.rect
            old_r = info.rect  # original position in old PDF

            # Normalised coords for new (right) side
            new_coords = {
                "x": new_r.x0 / tot_w,
                "y": new_r.y0 / ph,
                "w": (new_r.x1 - new_r.x0) / tot_w,
                "h": (new_r.y1 - new_r.y0) / ph,
            }
            dot_xy = None
            pin_side = 1
            if num_by_index.get(info.index) is not None:
                ddx, ddy = _dot_position(info, pw, ph, dot_r_pg, S_pg)
                dot_xy = (ddx / tot_w, ddy / ph)
                pin_side = _pin_side(info)
            # Normalised coords for old (left) side
            old_coords = {
                "x": old_r.x0 / old_pw,
                "y": old_r.y0 / old_ph,
                "w": (old_r.x1 - old_r.x0) / old_pw,
                "h": (old_r.y1 - old_r.y0) / old_ph,
            }

            overlay_annots.append({
                "index": info.index,
                "number": num_by_index.get(info.index),
                "page": info.page_num + 1,
                "type": info.annot_type[1],
                "status": info.status,
                "content": info.content,
                "author": info.author,
                "confidence": f"{info.match_confidence:.0%}",
                "method": info.match_method,
                "note": info.match_note,
                # New position (right/output side)
                "x": new_coords["x"],
                "y": new_coords["y"],
                "w": new_coords["w"],
                "h": new_coords["h"],
                # Original position (left/old side)
                "old_x": old_coords["x"],
                "old_y": old_coords["y"],
                "old_w": old_coords["w"],
                "old_h": old_coords["h"],
                "dot_x": dot_xy[0] if dot_xy else None,
                "dot_y": dot_xy[1] if dot_xy else None,
                "pin_side": pin_side,
            })
        pages_preview.append({
            "page_num": page_num + 1,
            "new_page_num": out_pn + 1,
            "width": tot_w,
            "content_frac": pw / tot_w,   # share of the output width that is page (rest = sidebar)
            "height": ph,
            "aspect": ph / tot_w,
            "old_width": old_pw,
            "old_height": old_ph,
            "old_aspect": old_ph / old_pw,
            "annotations": overlay_annots,
        })
    preview_doc.close()
    old_preview_doc.close()
    results["pages"] = pages_preview

    # ── 5. Write output PDF ───────────────────────────────────────────────────
    write_pdf(all_annots, new_pdf_path, output_path, skip_indices=set())
    # Preview copy WITHOUT page dots: the web view draws its own (draggable) markers
    preview_path = output_path[:-4] + "_preview.pdf" if output_path.endswith(".pdf") else output_path + "_preview"
    write_pdf(all_annots, new_pdf_path, preview_path, skip_indices=set(), draw_dots=False)
    results["_preview_path"] = preview_path
    old_doc.close()
    new_doc.close()

    # Store all_annots on results for use by confirm/re-write endpoint
    results["_all_annots"] = all_annots
    results["_new_pdf_path"] = new_pdf_path

    return results


def _wrap_text(text, max_chars):
    """Wrap text to lines of at most max_chars, breaking on spaces."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip() if current else word
    if current:
        lines.append(current)
    return lines


SIDEBAR_BASE = 200   # sidebar width at scale 1
DOT_BASE = 11        # dot radius at scale 1


def _scale(pw):
    """Sizes scale with page width (900pt baseline) so they stay legible on big pages."""
    return max(1.0, pw / 900.0)


PIN_L = 1.75   # pin length: distance from body centre to the point, in body radii


def _pin_side(info):
    """Which side of the point the pin body goes: the side the original callout's
    arrow came from (tip -> text box), i.e. where the reviewer put the box."""
    v = info.vertices or []
    if len(v) >= 2:
        return 1 if v[-1][0] >= v[0][0] else -1
    r = info.rect
    tip = _anchor_point(info)
    return 1 if (r.x0 + r.x1) / 2 >= tip[0] else -1


def _dot_position(info, pw, ph, dot_r, S, side=None):
    """Where the pin's POINT goes: the referenced spot, nudged 2*S toward the body
    so the point stops just short of the superscript instead of on it."""
    if side is None:
        side = _pin_side(info)
    x, y = _anchor_point(info)
    g = 2 * S / math.sqrt(2)
    return x + side * g, y - g


def _pin_body_center(px, py, r, side=1):
    """Body centre for a pin whose point is (px, py): up, and to `side`."""
    off = r * PIN_L / math.sqrt(2)
    return px + side * off, py - off


def _pin_polygon(cx, cy, px, py, r, steps=48):
    """Teardrop outline: point (px,py) + tangent lines + arc around the far side."""
    dx, dy = px - cx, py - cy
    d = math.hypot(dx, dy)
    if d < r * 1.15:                      # point is inside/at the body: plain circle
        return [(cx + r * math.cos(2 * math.pi * i / steps),
                 cy + r * math.sin(2 * math.pi * i / steps)) for i in range(steps)]
    th = math.atan2(dy, dx)
    phi = math.acos(r / d)
    a0, a1 = th + phi, th + 2 * math.pi - phi
    pts = [(cx + r * math.cos(a0 + (a1 - a0) * i / steps),
            cy + r * math.sin(a0 + (a1 - a0) * i / steps)) for i in range(steps + 1)]
    pts.append((px, py))
    return pts


def _draw_pin(page, cx, cy, px, py, number, radius):
    """Black teardrop pin with white number; point at (px,py), body centred (cx,cy)."""
    page.draw_polyline([fitz.Point(x, y) for x, y in _pin_polygon(cx, cy, px, py, radius)],
                       color=None, fill=(0, 0, 0), closePath=True)
    label = str(number)
    fontsize = radius * (1.1 if len(label) == 1 else 0.85)
    tw = fitz.get_text_length(label, fontname="helv", fontsize=fontsize)
    page.insert_text(fitz.Point(cx - tw / 2, cy + fontsize * 0.35), label,
                     fontname="helv", fontsize=fontsize, color=(1, 1, 1))


def _out_page(info):
    """Page of the NEW pdf this annotation belongs on."""
    return info.new_page_num if info.new_page_num is not None else info.page_num


def _anchor_point(info):
    """Where the dot goes: remapped tip point, else raw tip, else rect centre."""
    if info.new_vertices:
        return info.new_vertices[0]
    if info.tip_point is not None:
        return (info.tip_point.x, info.tip_point.y)
    r = info.new_rect or info.rect
    return (r.x0 + r.width / 2, r.y0 + r.height / 2)


def assign_numbers(all_annots):
    """Give every text (FreeText) annotation a stable global number in reading
    order (page, then top-to-bottom, left-to-right by dot position).
    Rectangles get no number. Returns {annot.index: number}."""
    numbered = [a for a in all_annots
                if a.annot_type[0] == ANNOT_FREETEXT and a.new_rect is not None]
    numbered.sort(key=lambda a: (_out_page(a), round(_anchor_point(a)[1] / 20), _anchor_point(a)[0]))
    return {a.index: n + 1 for n, a in enumerate(numbered)}


def _wrap_text_measured(text, max_w, fontsize):
    """Word-wrap text to a pixel width using real Helvetica metrics."""
    lines, current = [], ""
    for word in text.split():
        trial = (current + " " + word).strip()
        if current and fitz.get_text_length(trial, fontname="helv", fontsize=fontsize) > max_w:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _draw_number_dot(page, cx, cy, number, radius=7):
    """Filled black circle with centered white number at (cx, cy)."""
    page.draw_circle(fitz.Point(cx, cy), radius, color=(0, 0, 0), fill=(0, 0, 0))
    label = str(number)
    fontsize = radius * (1.1 if len(label) == 1 else 0.85)
    tw = fitz.get_text_length(label, fontname="helv", fontsize=fontsize)
    page.insert_text(fitz.Point(cx - tw / 2, cy + fontsize * 0.35), label,
                     fontname="helv", fontsize=fontsize, color=(1, 1, 1))


def write_pdf(all_annots, new_pdf_path, output_path, skip_indices=None,
              overrides=None, draw_dots=True, sides=None):
    """Write annotations to a new PDF with a numbered reference sidebar.

    Each page is expanded rightward by SIDEBAR_W points. Annotations get:
    - A filled black dot with white number at the tip/anchor point on the page
    - A numbered entry in the right sidebar listing the annotation content
    """
    if skip_indices is None:
        skip_indices = set()
    # overrides: {annot.index: (x_frac, y_frac)} — manual dot positions, as fractions of
    # the full output page (original page + sidebar) width and of page height.
    overrides = overrides or {}
    sides = sides or {}          # {annot.index: +1|-1} manual body side (else automatic)

    SIDEBAR_BG = (0.97, 0.97, 0.97)   # near-white background
    DIVIDER_COLOR = (0.8, 0.8, 0.8)   # light grey divider
    DOT_COLOR = (0.1, 0.1, 0.1)       # near-black dot

    out_doc = fitz.open(new_pdf_path)

    # ── Group active annotations by page ────────────────────────────────────
    active = [info for info in all_annots
              if info.index not in skip_indices and info.new_rect is not None]

    # Only text (FreeText) annotations get a number/dot/sidebar entry.
    # Numbers are assigned over ALL annotations so they stay stable when the
    # user removes some (gaps are fine; references never shift).
    num_by_index = assign_numbers(all_annots)
    numbered = [info for info in active
                if info.annot_type[0] == ANNOT_FREETEXT and info.index in num_by_index]
    numbered.sort(key=lambda a: num_by_index[a.index])
    global_num = {(_out_page(info), info.index): num_by_index[info.index] for info in numbered}

    # Group by page
    from collections import defaultdict
    page_annots = defaultdict(list)
    for info in numbered:
        page_annots[_out_page(info)].append(info)

    # ── Expand pages and draw ────────────────────────────────────────────────
    for page_num in range(len(out_doc)):
        page = out_doc[page_num]
        orig_rect = page.rect
        pw = orig_rect.width
        ph = orig_rect.height

        # Scale every size to the page width so it stays legible on large
        # pages (web-page PDFs are often ~1800pt wide vs 612pt for letter).
        S = _scale(pw)
        SIDEBAR_W = SIDEBAR_BASE * S
        SIDEBAR_PAD = 10 * S
        DOT_R = DOT_BASE * S
        HEADER_H = 26 * S
        annots_on_page = page_annots.get(page_num, [])

        # Expand the mediabox rightward
        new_mediabox = fitz.Rect(0, 0, pw + SIDEBAR_W, ph)
        page.set_mediabox(new_mediabox)
        page.set_cropbox(new_mediabox)


        # ── Draw sidebar background ──────────────────────────────────────────
        sidebar_rect = fitz.Rect(pw, 0, pw + SIDEBAR_W, ph)
        page.draw_rect(sidebar_rect, color=None, fill=SIDEBAR_BG)
        page.draw_line(fitz.Point(pw, 0), fitz.Point(pw, ph),
                       color=DIVIDER_COLOR, width=0.5 * S)

        # ── Pick the largest text size at which all entries fit the page ─────
        text_x_off = SIDEBAR_PAD + DOT_R * 2 + 6 * S
        text_w = SIDEBAR_W - text_x_off - SIDEBAR_PAD

        def _layout(font):
            lead = font * 1.3
            gap = 8 * S
            total = HEADER_H + SIDEBAR_PAD
            wrapped = []
            for inf in annots_on_page:
                lns = _wrap_text_measured(inf.content or "", text_w, font)
                wrapped.append(lns)
                total += max(DOT_R * 2, len(lns) * lead) + gap
            return wrapped, lead, gap, total

        font = 8 * S
        wrapped, lead, gap, total = _layout(font)
        while total > ph - SIDEBAR_PAD and font > 3:
            font *= 0.9
            wrapped, lead, gap, total = _layout(font)

        # "References" header
        if annots_on_page:
            hfs = 8 * S
            page.insert_text(fitz.Point(pw + SIDEBAR_PAD, SIDEBAR_PAD + hfs),
                             "References", fontname="helv", fontsize=hfs,
                             color=(0.4, 0.4, 0.4))
            rule_y = SIDEBAR_PAD + hfs + 5 * S
            page.draw_line(fitz.Point(pw + SIDEBAR_PAD, rule_y),
                           fitz.Point(pw + SIDEBAR_W - SIDEBAR_PAD, rule_y),
                           color=DIVIDER_COLOR, width=0.5 * S)

        # ── Draw sidebar entries ─────────────────────────────────────────────
        cursor_y = HEADER_H + SIDEBAR_PAD
        for info, lns in zip(annots_on_page, wrapped):
            num = global_num[(_out_page(info), info.index)]
            dot_x = pw + SIDEBAR_PAD + DOT_R
            dot_y = cursor_y + DOT_R
            _draw_number_dot(page, dot_x, dot_y, num, radius=DOT_R)

            text_y = cursor_y + font
            for line in lns:
                page.insert_text(fitz.Point(pw + text_x_off, text_y), line,
                                 fontname="helv", fontsize=font,
                                 color=(0.1, 0.1, 0.1))
                text_y += lead
            cursor_y += max(DOT_R * 2, len(lns) * lead) + gap

        # ── Draw numbered dots on page content ──────────────────────────────
        for info in annots_on_page:
            num = global_num[(_out_page(info), info.index)]

            if not draw_dots:
                continue
            side = sides.get(info.index) or _pin_side(info)
            if info.index in overrides:
                # Manually placed: the pin's POINT goes exactly where it was dropped
                fx, fy = overrides[info.index]
                px, py = fx * (pw + SIDEBAR_W), fy * ph
            else:
                px, py = _dot_position(info, pw, ph, DOT_R, S, side)

            # Keep the body on the page; the point stays on the target
            cx, cy = _pin_body_center(px, py, DOT_R, side)
            cx = max(DOT_R + 1, min(cx, pw - DOT_R - 1))
            cy = max(DOT_R + 1, min(cy, ph - DOT_R - 1))
            _draw_pin(page, cx, cy, px, py, num, DOT_R)

    # Rectangles are intentionally not written: the numbered dots replace them.

    out_doc.save(output_path, garbage=4, deflate=True)
    out_doc.close()

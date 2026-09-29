"""
PDF Annotation Remapper v2
Remaps callout and rectangle annotations from an old PDF to a new version,
preserving metadata (author, dates, reply threads, review status).
"""

import fitz  # PyMuPDF
import re
from dataclasses import dataclass, field
from typing import Optional


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
    status: str = "needs_look"  # "moved" | "needs_look" | "content_removed"
    new_rect: Optional[fitz.Rect] = None
    new_vertices: Optional[list] = None
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
        "content_removed": 0,
        "annotations": []
    }

    # ── 1. Extract all annotations from old doc ──────────────────────────────
    all_annots = []
    for page_num in range(len(old_doc)):
        page = old_doc[page_num]
        for i, annot in enumerate(page.annots()):
            atype = annot.type  # (int, name)
            # Skip popups — they travel with their parent annotation
            if atype[0] == ANNOT_POPUP:
                continue

            vertices = list(annot.vertices or [])
            tip_point = fitz.Point(vertices[0][0], vertices[0][1]) if vertices else None

            annot_info_dict = annot.info
            colors = annot.colors  # {"stroke": rgb_tuple, "fill": rgb_tuple}
            border = annot.border  # {"width": float, "style": str}

            info = AnnotationInfo(
                index=i,
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
            )

            # ── Fingerprint strategy ──────────────────────────────────────────
            # For rectangles: use text INSIDE the rect as fingerprint
            if atype[0] == ANNOT_SQUARE:
                rect_text = extract_text_in_rect(page, info.rect)
                if rect_text and len(rect_text) > 4:
                    info.fingerprint_text = rect_text
                    info.fingerprint_rect = fitz.Rect(info.rect)
                    ctx_b, ctx_a = get_words_around_rect(page, info.rect)
                    info.context_before = ctx_b
                    info.context_after = ctx_a

            # For callouts: prefer annotation content text (REF strings etc.)
            if not info.fingerprint_text:
                content_text = info.content.strip()
                if content_text and len(content_text) > 8:
                    info.fingerprint_text = content_text
                    info.fingerprint_rect = None
                elif tip_point:
                    fp_text, fp_rect = extract_text_near_point(page, tip_point)
                    if not fp_text and vertices and len(vertices) >= 2:
                        anchor = fitz.Point(vertices[-1][0], vertices[-1][1])
                        fp_text, fp_rect = extract_text_near_point(page, anchor)
                    info.fingerprint_text = fp_text
                    info.fingerprint_rect = fp_rect
                    if fp_rect:
                        ctx_b, ctx_a = get_words_around_rect(page, fp_rect)
                        info.context_before = ctx_b
                        info.context_after = ctx_a

            all_annots.append(info)

    results["total"] = len(all_annots)

    # ── 2. Match each annotation in the new doc ───────────────────────────────
    for info in all_annots:
        key_words = extract_key_words(info.fingerprint_text) if info.fingerprint_text else []

        # Check if the content still exists anywhere in the new doc
        if info.fingerprint_text and key_words:
            still_exists = text_exists_anywhere(new_doc, info.fingerprint_text, key_words)
        else:
            still_exists = True  # can't tell, assume yes

        # Try same page first, then other pages
        target_pages = []
        if info.page_num < len(new_doc):
            target_pages.append(info.page_num)
        target_pages += [p for p in range(len(new_doc)) if p != info.page_num]

        for new_page_num in target_pages:
            new_page = new_doc[new_page_num]

            if info.fingerprint_text and key_words:
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
                    info.status = "moved" if (abs(offset.x) > 2 or abs(offset.y) > 2) else "moved"
                    info.match_note = (f"Moved to page {new_page_num + 1} "
                                       f"(confidence {confidence:.0%})")
                    break
            else:
                # No fingerprint — keep original position
                if new_page_num == info.page_num:
                    info.matched = True
                    info.match_confidence = 0.5
                    info.new_rect = fitz.Rect(info.rect)
                    info.new_vertices = list(info.vertices)
                    info.status = "needs_look"
                    info.match_note = "No text anchor — kept original position"
                    break

        # ── Classify unmatched ────────────────────────────────────────────────
        if not info.matched:
            if not still_exists:
                info.status = "content_removed"
                info.match_note = "Annotated content no longer exists in new PDF"
                results["content_removed"] += 1
            else:
                info.status = "needs_look"
                info.match_note = "Content may exist but position could not be determined"
            # Keep at original position so nothing is silently lost
            if info.page_num < len(new_doc):
                info.new_rect = fitz.Rect(info.rect)
                info.new_vertices = list(info.vertices)

        # Low-confidence matches also need a look
        if info.matched and info.match_confidence < 0.4:
            info.status = "needs_look"

        results["matched" if info.matched else "unmatched"] += 1
        results["annotations"].append({
            "index": info.index,
            "page": info.page_num + 1,
            "type": info.annot_type[1],
            "content": info.content[:80] + ("..." if len(info.content) > 80 else ""),
            "author": info.author,
            "status": info.status,
            "confidence": f"{info.match_confidence:.0%}",
            "note": info.match_note,
        })

    # ── 3. Build page preview data (dimensions + annotation overlays) ─────────
    # Group annotations by page for the preview renderer
    annots_by_page = {}
    for info in all_annots:
        pg = info.page_num
        if pg not in annots_by_page:
            annots_by_page[pg] = []
        annots_by_page[pg].append(info)

    pages_preview = []
    preview_doc = fitz.open(new_pdf_path)
    for page_num in sorted(annots_by_page.keys()):
        if page_num >= len(preview_doc):
            continue
        page = preview_doc[page_num]
        pw, ph = page.rect.width, page.rect.height
        overlay_annots = []
        for info in annots_by_page[page_num]:
            r = info.new_rect if info.new_rect else info.rect
            overlay_annots.append({
                "index": info.index,
                "status": info.status,
                "content": info.content[:120] + ("…" if len(info.content) > 120 else ""),
                "author": info.author,
                "confidence": f"{info.match_confidence:.0%}",
                "note": info.match_note,
                # Normalised coords (0–1) so the frontend can scale to any width
                "x": r.x0 / pw,
                "y": r.y0 / ph,
                "w": (r.x1 - r.x0) / pw,
                "h": (r.y1 - r.y0) / ph,
            })
        pages_preview.append({
            "page_num": page_num + 1,
            "width": pw,
            "height": ph,
            "aspect": ph / pw,
            "annotations": overlay_annots,
        })
    preview_doc.close()
    results["pages"] = pages_preview

    # ── 4. Write output PDF ───────────────────────────────────────────────────
    out_doc = fitz.open(new_pdf_path)

    for info in all_annots:
        if info.new_rect is None:
            continue
        target_page_num = info.page_num
        if target_page_num >= len(out_doc):
            continue
        out_page = out_doc[target_page_num]

        meta = {
            "author": info.author,
            "content": info.content,
            "subject": info.subject,
            "creation_date": info.creation_date,
            "mod_date": info.mod_date,
        }

        if info.annot_type[0] == ANNOT_FREETEXT:
            fill = info.colors.get("fill") or (1, 1, 0.6)
            text_color = info.colors.get("stroke") or (0, 0, 0)
            annot = out_page.add_freetext_annot(
                info.new_rect, info.content,
                fontsize=7,
                text_color=text_color,
                fill_color=fill,
            )
            annot.set_rect(info.new_rect)
            copy_annot_metadata(meta, annot)
            annot.update()

        elif info.annot_type[0] == ANNOT_SQUARE:
            annot = out_page.add_rect_annot(info.new_rect)
            stroke = info.colors.get("stroke") or (1, 0, 0)
            fill   = info.colors.get("fill")
            color_dict = {"stroke": stroke}
            if fill:
                color_dict["fill"] = fill
            annot.set_colors(color_dict)
            bw = info.border.get("width", 2) if info.border else 2
            annot.set_border(width=bw)
            copy_annot_metadata(meta, annot)
            annot.update()

    out_doc.save(output_path, garbage=4, deflate=True)
    old_doc.close()
    new_doc.close()
    out_doc.close()

    return results

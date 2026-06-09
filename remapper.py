"""
PDF Annotation Remapper
Remaps callout annotations from an old PDF to a new version by finding
where the annotated content moved and updating coordinates accordingly.
"""

import fitz  # PyMuPDF
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnnotationInfo:
    index: int
    page_num: int
    annot_type: tuple
    rect: fitz.Rect
    content: str
    vertices: list
    flags: int
    # Fingerprint: text near the arrow tip in the original PDF
    tip_point: Optional[fitz.Point] = None
    fingerprint_text: str = ""
    fingerprint_rect: Optional[fitz.Rect] = None
    # Result
    matched: bool = False
    match_confidence: float = 0.0
    new_rect: Optional[fitz.Rect] = None
    new_vertices: Optional[list] = None
    match_note: str = ""


def extract_text_near_point(page: fitz.Page, point: fitz.Point, radius: float = 80) -> tuple[str, fitz.Rect]:
    """Extract text in a region around a point, expanding if needed."""
    search_rect = fitz.Rect(
        point.x - radius, point.y - radius,
        point.x + radius, point.y + radius
    )
    words = page.get_text("words", clip=search_rect)
    if not words and radius < 300:
        return extract_text_near_point(page, point, radius * 2)
    text = " ".join(w[4] for w in words)
    if words:
        actual_rect = fitz.Rect(
            min(w[0] for w in words), min(w[1] for w in words),
            max(w[2] for w in words), max(w[3] for w in words)
        )
    else:
        actual_rect = search_rect
    return text.strip(), actual_rect


def clean_fingerprint(text: str) -> str:
    """Normalize text for matching: lowercase, collapse spaces, strip punctuation noise."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s%(),.-]', '', text)
    return text.strip()


def extract_key_words(text: str, min_len: int = 4) -> list[str]:
    """Extract meaningful words for matching, filtering short/common words."""
    stopwords = {'the', 'and', 'for', 'are', 'was', 'with', 'this', 'that', 'from', 'have', 'has'}
    words = clean_fingerprint(text).split()
    return [w for w in words if len(w) >= min_len and w not in stopwords]


def find_text_in_page(page: fitz.Page, fingerprint_text: str, key_words: list[str]) -> tuple[Optional[fitz.Rect], float]:
    """
    Search for fingerprint text in a page.
    Returns (rect_of_match, confidence_score).
    """
    if not key_words:
        return None, 0.0

    # Strategy 1: Search for longer phrase chunks (most reliable)
    words = fingerprint_text.split()
    best_rect = None
    best_score = 0.0

    # Try progressively shorter phrase windows
    for window in [6, 4, 3, 2]:
        for i in range(len(words) - window + 1):
            phrase = " ".join(words[i:i+window])
            if len(phrase) < 8:
                continue
            hits = page.search_for(phrase, quads=False)
            if hits:
                score = window / max(len(words), 1)
                score = min(score, 1.0)
                if score > best_score:
                    best_score = score
                    best_rect = hits[0]
        if best_rect and best_score > 0.5:
            break

    if best_rect:
        return best_rect, best_score

    # Strategy 2: Search for individual key words, find cluster
    hit_rects = []
    for word in key_words[:8]:  # limit to first 8 key words
        hits = page.search_for(word)
        if hits:
            hit_rects.extend(hits)

    if not hit_rects:
        return None, 0.0

    # Find the tightest cluster of hits
    # Group hits that are close together (within 200pt vertically)
    if len(hit_rects) == 1:
        return hit_rects[0], 0.3

    best_cluster_rect = None
    best_cluster_score = 0.0
    for i, anchor in enumerate(hit_rects):
        nearby = [r for r in hit_rects if abs(r.y0 - anchor.y0) < 200 and abs(r.x0 - anchor.x0) < 400]
        if len(nearby) >= 2:
            cluster_rect = fitz.Rect(
                min(r.x0 for r in nearby), min(r.y0 for r in nearby),
                max(r.x1 for r in nearby), max(r.y1 for r in nearby)
            )
            score = len(nearby) / len(key_words)
            if score > best_cluster_score:
                best_cluster_score = score
                best_cluster_rect = cluster_rect

    if best_cluster_rect:
        return best_cluster_rect, min(best_cluster_score, 0.7)

    return hit_rects[0], 0.2


def remap_annotation(annot_info: AnnotationInfo, offset: fitz.Point) -> tuple[fitz.Rect, list]:
    """Apply an offset to annotation rect and vertices."""
    new_rect = fitz.Rect(
        annot_info.rect.x0 + offset.x,
        annot_info.rect.y0 + offset.y,
        annot_info.rect.x1 + offset.x,
        annot_info.rect.y1 + offset.y,
    )
    new_vertices = [(v[0] + offset.x, v[1] + offset.y) for v in annot_info.vertices]
    return new_rect, new_vertices


def process_pdfs(old_pdf_path: str, new_pdf_path: str, output_path: str) -> dict:
    """
    Main function: extract annotations from old PDF, find matches in new PDF,
    write remapped annotations to output PDF.
    Returns a results dict with per-annotation match info.
    """
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)

    results = {
        "total": 0,
        "matched": 0,
        "unmatched": 0,
        "annotations": []
    }

    # Collect all annotations from old PDF
    all_annots: list[AnnotationInfo] = []
    for page_num in range(len(old_doc)):
        page = old_doc[page_num]
        for i, annot in enumerate(page.annots()):
            vertices = annot.vertices or []
            tip_point = None
            if vertices:
                # For callouts: tip is the first vertex (arrow head)
                tip_point = fitz.Point(vertices[0][0], vertices[0][1])

            info = AnnotationInfo(
                index=i,
                page_num=page_num,
                annot_type=annot.type,
                rect=fitz.Rect(annot.rect),
                content=annot.info.get("content", ""),
                vertices=list(vertices),
                flags=annot.flags,
                tip_point=tip_point
            )

            # Build fingerprint: prefer the annotation's own content text,
            # then fall back to text near the tip/anchor points
            content_text = annot.info.get("content", "").strip()
            if content_text and len(content_text) > 8:
                info.fingerprint_text = content_text
                info.fingerprint_rect = None  # will be resolved at match time
            elif tip_point:
                fp_text, fp_rect = extract_text_near_point(page, tip_point)
                # Also try anchor point (last vertex) if tip has no text
                if not fp_text and vertices and len(vertices) >= 2:
                    anchor = fitz.Point(vertices[-1][0], vertices[-1][1])
                    fp_text, fp_rect = extract_text_near_point(page, anchor)
                info.fingerprint_text = fp_text
                info.fingerprint_rect = fp_rect

            all_annots.append(info)

    results["total"] = len(all_annots)

    # Match each annotation in the new PDF
    for info in all_annots:
        # Find the corresponding page in the new doc
        # Try same page first, then search all pages
        target_pages = []
        if info.page_num < len(new_doc):
            target_pages.append(info.page_num)
        target_pages += [p for p in range(len(new_doc)) if p != info.page_num]

        matched = False
        for new_page_num in target_pages:
            new_page = new_doc[new_page_num]

            if info.fingerprint_text:
                key_words = extract_key_words(info.fingerprint_text)
                match_rect, confidence = find_text_in_page(new_page, info.fingerprint_text, key_words)

                if match_rect and confidence > 0.15:
                    # Compute offset from old content location → new content location
                    if info.fingerprint_rect:
                        old_center = fitz.Point(
                            (info.fingerprint_rect.x0 + info.fingerprint_rect.x1) / 2,
                            (info.fingerprint_rect.y0 + info.fingerprint_rect.y1) / 2
                        )
                    else:
                        # Content-based fingerprint: find it in the OLD doc too
                        old_page = old_doc[info.page_num]
                        old_match, _ = find_text_in_page(old_page, info.fingerprint_text, key_words)
                        if old_match:
                            old_center = fitz.Point(
                                (old_match.x0 + old_match.x1) / 2,
                                (old_match.y0 + old_match.y1) / 2
                            )
                        else:
                            # Fall back to annotation rect center
                            old_center = fitz.Point(
                                (info.rect.x0 + info.rect.x1) / 2,
                                (info.rect.y0 + info.rect.y1) / 2
                            )

                    new_center = fitz.Point(
                        (match_rect.x0 + match_rect.x1) / 2,
                        (match_rect.y0 + match_rect.y1) / 2
                    )
                    offset = fitz.Point(new_center.x - old_center.x, new_center.y - old_center.y)

                    new_rect, new_vertices = remap_annotation(info, offset)
                    info.matched = True
                    info.match_confidence = confidence
                    info.new_rect = new_rect
                    info.new_vertices = new_vertices
                    info.match_note = f"Matched on page {new_page_num + 1} (confidence: {confidence:.0%})"
                    matched = True
                    break
            else:
                # Square/Rectangle with no text — keep same position if page exists
                if new_page_num == info.page_num:
                    info.matched = True
                    info.match_confidence = 0.5
                    info.new_rect = fitz.Rect(info.rect)
                    info.new_vertices = list(info.vertices)
                    info.match_note = "No text anchor — kept original position"
                    matched = True
                    break

        if not matched:
            info.match_note = "No match found — annotation not placed"

        if info.matched:
            results["matched"] += 1
        else:
            results["unmatched"] += 1

        results["annotations"].append({
            "index": info.index,
            "page": info.page_num + 1,
            "type": info.annot_type[1],
            "content": info.content[:80] + ("..." if len(info.content) > 80 else ""),
            "matched": info.matched,
            "confidence": f"{info.match_confidence:.0%}",
            "note": info.match_note
        })

    # Write output PDF: copy new PDF and write remapped annotations
    out_doc = fitz.open(new_pdf_path)

    for info in all_annots:
        if not info.matched or info.new_rect is None:
            continue

        target_page_num = info.page_num
        # Find which page the match ended up on
        for a in results["annotations"]:
            if a["index"] == info.index and "page" in a:
                # page stored as 1-indexed
                pass

        if target_page_num >= len(out_doc):
            continue

        out_page = out_doc[target_page_num]

        if info.annot_type[0] == 2:  # FreeText / Callout
            annot = out_page.add_freetext_annot(
                info.new_rect,
                info.content,
                fontsize=7,
                text_color=(0, 0, 0),
                fill_color=(1, 1, 0.6),
            )
            # Write callout vertices directly via xref update
            if info.new_vertices and len(info.new_vertices) >= 2:
                # Build the PDF CL (callout line) array string
                pts = info.new_vertices
                cl_values = " ".join(f"{v[0]:.2f} {v[1]:.2f}" for v in pts)
                # Update via set_rect to trigger appearance rebuild
                annot.set_rect(info.new_rect)
            annot.update()

        elif info.annot_type[0] == 4:  # Square
            annot = out_page.add_rect_annot(info.new_rect)
            annot.set_colors(stroke=(1, 0, 0))
            annot.set_border(width=2)
            annot.update()

    out_doc.save(output_path, garbage=4, deflate=True)
    old_doc.close()
    new_doc.close()
    out_doc.close()

    return results

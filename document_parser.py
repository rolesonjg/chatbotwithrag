from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import re

import pymupdf


# ---------------------------------------------------------
# Text utilities
# ---------------------------------------------------------

def norm(text: str) -> str:
    """Normalize whitespace while preserving readable text."""
    return re.sub(r"[ \t]+", " ", text or "").strip()


def normalize_heading(text: str) -> str:
    """Normalize whitespace and line breaks in a heading."""
    return re.sub(r"\s+", " ", text or "").strip()


def extract_blocks(page):
    """
    Extract text blocks and basic typography from a PDF page.

    Each block contains:
      - text
      - bounding box
      - maximum font size
      - proportion of bold characters
      - number of text lines
    """
    output = []
    page_dict = page.get_text("dict", sort=True)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        lines = []
        sizes = []
        bold_chars = 0
        total_chars = 0

        for line in block.get("lines", []):
            line_text = norm(
                "".join(
                    span.get("text", "")
                    for span in line.get("spans", [])
                )
            )

            if line_text:
                lines.append(line_text)

            for span in line.get("spans", []):
                text = span.get("text", "").strip()

                if not text:
                    continue

                sizes.append(float(span.get("size", 0)))
                total_chars += len(text)

                font = span.get("font", "").lower()

                if any(
                    name in font
                    for name in ("bold", "black", "demi", "heavy")
                ):
                    bold_chars += len(text)

        text = "\n".join(lines).strip()

        if not text:
            continue

        output.append(
            {
                "text": text,
                "bbox": block.get("bbox", [0, 0, 0, 0]),
                "font_size": max(sizes) if sizes else 0,
                "bold_ratio": (
                    bold_chars / total_chars
                    if total_chars
                    else 0
                ),
                "line_count": len(lines),
            }
        )

    return output


# ---------------------------------------------------------
# Heading detection
# ---------------------------------------------------------

COMMON_HEADINGS = {
    "abstract",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "authority",
    "bibliography",
    "contents",
    "conclusion",
    "epilogue",
    "foreword",
    "glossary",
    "index",
    "introduction",
    "interpretation",
    "observance",
    "preface",
    "prologue",
    "references",
    "reversal",
    "summary",
}


def is_title_like(text: str) -> bool:
    """
    Identify plausible headings using text patterns.

    Typography is evaluated separately in parse_pdf().
    """
    text = normalize_heading(text)

    if len(text) < 4 or len(text) > 140:
        return False

    words = text.split()

    if len(words) > 14:
        return False

    # Reject isolated decorative letters.
    if re.fullmatch(r"[A-Za-z]", text):
        return False

    # Generic labels alone do not identify a section.
    if re.fullmatch(r"LAW", text, re.IGNORECASE):
        return False

    # Reject sentence-like blocks.
    if text.endswith((".", ",", ";", ":")):
        return False

    # Common document hierarchy patterns.
    if re.match(
        r"^(chapter|section|part|appendix|unit)"
        r"\s+[\w.-]+",
        text,
        re.IGNORECASE,
    ):
        return True

    # Numbered headings, including numbered laws.
    if re.match(r"^law\s+\d+\b", text, re.IGNORECASE):
        return True

    if re.match(
        r"^\d+(?:\.\d+){0,4}\.?\s+\S+",
        text,
    ):
        return True

    # Known generic section labels.
    if text.casefold() in COMMON_HEADINGS:
        return True

    # Uppercase headings should contain multiple words.
    letters = [char for char in text if char.isalpha()]

    if len(words) >= 2 and letters:
        uppercase_ratio = (
            sum(char.isupper() for char in letters)
            / len(letters)
        )

        if uppercase_ratio >= 0.78:
            return True

    return False


def is_heading(block: dict, body_size: float) -> bool:
    """
    Decide whether a block is a heading.

    Text patterns and typography must provide supporting evidence.
    Bold or large text alone is not enough.
    """
    text = block["text"].strip()
    normalized = normalize_heading(text)

    if not is_title_like(normalized):
        return False

    if block.get("line_count", 1) > 3:
        return False

    if len(normalized.split()) > 14:
        return False

    # Strong structural patterns can identify headings without
    # relying on font size.
    strong_pattern = bool(
        re.match(
            r"^(chapter|section|part|appendix|unit)"
            r"\s+[\w.-]+",
            normalized,
            re.IGNORECASE,
        )
        or re.match(
            r"^law\s+\d+\b",
            normalized,
            re.IGNORECASE,
        )
        or re.match(
            r"^\d+(?:\.\d+){0,4}\.?\s+\S+",
            normalized,
        )
    )

    if strong_pattern:
        return True

    # Recognize single-word section headings only when
    # they match the known heading vocabulary.
    if normalized.casefold() in COMMON_HEADINGS:
        return True

    letters = [char for char in normalized if char.isalpha()]

    uppercase_heading = (
        len(normalized.split()) >= 2
        and bool(letters)
        and (
            sum(char.isupper() for char in letters)
            / len(letters)
        ) >= 0.78
    )

    large_font = (
        body_size > 0
        and block["font_size"] >= body_size * 1.18
    )

    bold_text = block["bold_ratio"] >= 0.72

    # Typography supports a heading only when the text itself
    # already looks like a plausible title.
    return (
        uppercase_heading
        or (
            (large_font or bold_text)
            and len(normalized.split()) >= 2
        )
    )


# ---------------------------------------------------------
# Repeated header/footer removal
# ---------------------------------------------------------

def remove_repeated_marginalia(pages):
    """
    Remove repeated text blocks near the top or bottom of pages.

    The page's actual dimensions are used instead of assuming
    every PDF page is US Letter size.
    """
    counts = Counter()
    page_total = len(pages)

    for blocks in pages:
        for block in blocks:
            text = normalize_heading(block["text"]).casefold()

            if not text or len(text) > 100:
                continue

            y0, y1 = block["bbox"][1], block["bbox"][3]

            # Estimate page boundaries from each block's stored
            # page dimensions, supplied below by parse_pdf().
            page_height = block.get("page_height", 792)

            near_margin = (
                y0 < page_height * 0.10
                or y1 > page_height * 0.90
            )

            if near_margin:
                counts[text] += 1

    threshold = max(3, int(page_total * 0.08))

    repeated = {
        text
        for text, count in counts.items()
        if count >= threshold
    }

    for blocks in pages:
        blocks[:] = [
            block
            for block in blocks
            if not (
                normalize_heading(block["text"]).casefold()
                in repeated
                and (
                    block["bbox"][1]
                    < block.get("page_height", 792) * 0.10
                    or block["bbox"][3]
                    > block.get("page_height", 792) * 0.90
                )
            )
        ]


# ---------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------

def parse_pdf(path: Path):
    """
    Parse a PDF into section-oriented records.

    Returns records containing:
      document_id, document_name, source,
      section_title, section_path,
      page_start, page_end, content, record_type.

    This uses lightweight heuristics and PyMuPDF rather than
    a heavyweight document-layout model.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path.resolve())

    pages = []

    with pymupdf.open(path) as pdf:
        if len(pdf) == 0:
            return []

        for page in pdf:
            page_blocks = extract_blocks(page)
            page_height = float(page.rect.height)

            for block in page_blocks:
                block["page_height"] = page_height

            pages.append(page_blocks)

    remove_repeated_marginalia(pages)

    sizes = [
        block["font_size"]
        for page_blocks in pages
        for block in page_blocks
        if block["font_size"] > 0
    ]

    body_size = (
        sorted(sizes)[len(sizes) // 2]
        if sizes
        else 10.0
    )

    document_id = hashlib.sha1(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:12]

    records = []
    section_stack = []

    current_heading = "Document body"
    current_content = []
    current_pages = []

    def flush():
        nonlocal current_content, current_pages

        content = "\n\n".join(
            item for item in current_content
            if item.strip()
        ).strip()

        if content and current_pages:
            records.append(
                {
                    "document_id": document_id,
                    "document_name": path.name,
                    "source": str(path.resolve()),
                    "section_title": current_heading,
                    "section_path": (
                        " > ".join(section_stack)
                        if section_stack
                        else "Document body"
                    ),
                    "page_start": min(current_pages),
                    "page_end": max(current_pages),
                    "content": content,
                    "record_type": (
                        "section"
                        if current_heading != "Document body"
                        else "body"
                    ),
                }
            )

        current_content = []
        current_pages = []

    for page_number, blocks in enumerate(pages, start=1):
        for block in blocks:
            text = block["text"].strip()

            if not text:
                continue

            heading = is_heading(block, body_size)

            if heading:
                normalized = normalize_heading(text)

                # Avoid repeating the same heading when it is
                # duplicated by adjacent PDF text blocks.
                if (
                    normalized.casefold()
                    == current_heading.casefold()
                    and not current_content
                ):
                    continue

                flush()

                current_heading = normalized

                # Major document divisions establish a new root.
                if re.match(
                    r"^(chapter|part|unit|appendix)\b",
                    normalized,
                    re.IGNORECASE,
                ):
                    section_stack = [normalized]

                # Other headings are nested under the current
                # major division where one exists.
                elif section_stack:
                    section_stack = (
                        section_stack[:1] + [normalized]
                    )

                else:
                    section_stack = [normalized]

                current_content.append(normalized)
                current_pages.append(page_number)

            else:
                current_content.append(text)
                current_pages.append(page_number)

    flush()

    # Fallback: if no usable blocks were extracted, use one
    # record per non-empty page.
    if not records:
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text", sort=True).strip()

                if not text:
                    continue

                records.append(
                    {
                        "document_id": document_id,
                        "document_name": path.name,
                        "source": str(path.resolve()),
                        "section_title": "Document body",
                        "section_path": "Document body",
                        "page_start": page_number,
                        "page_end": page_number,
                        "content": text,
                        "record_type": "body",
                    }
                )

    return records

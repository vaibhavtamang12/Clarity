"""Programmatic fixture builders — no binary files committed to the repo.

PDFs and DOCX fixtures are generated at test time with the same libraries
the parsers use, which keeps tests deterministic and self-describing.
"""

from __future__ import annotations

import io

import pymupdf
from docx import Document as DocxDocument


def build_pdf(pages: list[list[tuple[str, int, bool]]]) -> bytes:
    """pages: list of pages; each page is a list of (text, font_size, bold)."""
    doc = pymupdf.open()
    for page_lines in pages:
        page = doc.new_page()
        y = 72.0
        for text, size, bold in page_lines:
            page.insert_text(
                (50, y), text, fontsize=size, fontname="hebo" if bold else "helv"
            )
            y += size + 14
    data = doc.tobytes()
    doc.close()
    return data


def build_docx(items: list[tuple[str, str]]) -> bytes:
    """items: list of (style_name, text), e.g. ("Heading 1", "Policy")."""
    document = DocxDocument()
    for style, text in items:
        document.add_paragraph(text, style=style)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


SAMPLE_MARKDOWN = """# Refund Policy

Enterprise customers can request a refund within 30 days.

## Conditions

- The subscription must be unused.
- Refunds require written notice.

## Exclusions

Custom contracts are excluded.
"""

SAMPLE_HTML = """<!doctype html>
<html>
<head><title>Refund Policy</title><meta name="author" content="Legal Team"></head>
<body>
<header>Site header to be stripped</header>
<script>var x = 1;</script>
<h1>Refund Policy</h1>
<p>Enterprise customers can request a refund within 30 days.</p>
<h2>Conditions</h2>
<p>The subscription must be unused.</p>
</body>
</html>
"""

SAMPLE_TXT = """REFUND POLICY

Enterprise customers can request a refund within 30 days.

CONDITIONS

The subscription must be unused. Refunds require written notice.
"""
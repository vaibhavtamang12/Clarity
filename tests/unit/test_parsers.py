"""Unit tests for all five parsers (fixtures generated, not committed)."""

from __future__ import annotations

from app.ingestion.domain import BlockType
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.html_parser import HtmlParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.txt_parser import TxtParser
from tests.fixtures.builders import (
    SAMPLE_HTML,
    SAMPLE_MARKDOWN,
    SAMPLE_TXT,
    build_docx,
    build_pdf,
)


def test_txt_parser_detects_headings() -> None:
    output = TxtParser().parse(SAMPLE_TXT.encode(), source_uri="policy.txt")
    headings = [b for b in output.blocks if b.is_heading]
    assert [h.text for h in headings] == ["REFUND POLICY", "CONDITIONS"]
    assert output.metadata.page_count == 1


def test_markdown_parser_structure() -> None:
    output = MarkdownParser().parse(SAMPLE_MARKDOWN.encode(), source_uri="policy.md")
    headings = [(b.text, b.heading_level) for b in output.blocks if b.is_heading]
    assert headings == [("Refund Policy", 1), ("Conditions", 2), ("Exclusions", 2)]
    list_items = [b for b in output.blocks if b.block_type == BlockType.LIST_ITEM]
    assert len(list_items) == 2
    assert output.metadata.title == "Refund Policy"


def test_html_parser_strips_noise_and_keeps_content() -> None:
    output = HtmlParser().parse(SAMPLE_HTML.encode(), source_uri="policy.html")
    all_text = " ".join(b.text for b in output.blocks)
    assert "var x" not in all_text          # script stripped
    assert "Site header" not in all_text    # header stripped
    assert "Enterprise customers" in all_text
    assert output.metadata.title == "Refund Policy"
    assert output.metadata.author == "Legal Team"


def test_pdf_parser_pages_and_headings() -> None:
    content = build_pdf(
        [
            [("Refund Policy", 18, True), ("Enterprise customers can request refunds.", 11, False)],
            [("Conditions", 18, True), ("The subscription must be unused.", 11, False)],
        ]
    )
    output = PdfParser().parse(content, source_uri="policy.pdf")
    pages = {b.page for b in output.blocks}
    assert pages == {1, 2}
    headings = [b.text for b in output.blocks if b.is_heading]
    assert headings == ["Refund Policy", "Conditions"]
    assert output.metadata.page_count == 2


def test_docx_parser_heading_levels() -> None:
    content = build_docx(
        [
            ("Heading 1", "Refund Policy"),
            ("Normal", "Enterprise customers can request a refund."),
            ("Heading 2", "Conditions"),
            ("Normal", "The subscription must be unused."),
        ]
    )
    output = DocxParser().parse(content, source_uri="policy.docx")
    headings = [(b.text, b.heading_level) for b in output.blocks if b.is_heading]
    assert headings == [("Refund Policy", 1), ("Conditions", 2)]
    assert output.metadata.title == "Refund Policy"
"""HTML parser for web-page ingestion.

Strips non-content regions (script/style/nav/footer/header/aside) before
walking the DOM. Title comes from <title> or og:title; author from the
`author` meta tag when present.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from app.ingestion.domain import (
    BlockType,
    ExtractedMetadata,
    ParserOutput,
    StructuralBlock,
)
from app.ingestion.parsers.base import Parser
from app.models.enums import DocumentSourceType

_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript", "form")
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


class HtmlParser(Parser):
    source_type = DocumentSourceType.HTML
    name = "html"

    def parse(self, content: bytes, *, source_uri: str) -> ParserOutput:
        soup = BeautifulSoup(content, "html.parser")

        for tag_name in _STRIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        title: str | None = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            og = soup.find("meta", attrs={"property": "og:title"})
            if og and og.get("content"):
                title = str(og["content"]).strip()

        author: str | None = None
        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag and author_tag.get("content"):
            author = str(author_tag["content"]).strip()

        root = soup.body or soup
        blocks: list[StructuralBlock] = []
        for element in root.find_all(list(_HEADINGS) + ["p", "li", "pre"]):
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            if element.name in _HEADINGS:
                blocks.append(
                    StructuralBlock(
                        text=text,
                        block_type=BlockType.HEADING,
                        heading_level=_HEADINGS[element.name],
                        page=1,
                    )
                )
            elif element.name == "pre":
                blocks.append(StructuralBlock(text=text, block_type=BlockType.CODE, page=1))
            elif element.name == "li":
                blocks.append(StructuralBlock(text=text, block_type=BlockType.LIST_ITEM, page=1))
            else:
                blocks.append(StructuralBlock(text=text, page=1))

        return ParserOutput(
            blocks=blocks, metadata=ExtractedMetadata(title=title, author=author, page_count=1)
        )
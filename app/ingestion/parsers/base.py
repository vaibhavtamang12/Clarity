"""Parser interface + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.exceptions import UnsupportedDocumentTypeError
from app.ingestion.domain import ParserOutput
from app.models.enums import DocumentSourceType


class Parser(ABC):
    """Converts raw bytes of one format into the shared block IR."""

    source_type: DocumentSourceType
    name: str

    @abstractmethod
    def parse(self, content: bytes, *, source_uri: str) -> ParserOutput: ...


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[DocumentSourceType, Parser] = {}

    def register(self, parser: Parser) -> None:
        self._parsers[parser.source_type] = parser

    def get(self, source_type: DocumentSourceType) -> Parser:
        parser = self._parsers.get(source_type)
        if parser is None:
            raise UnsupportedDocumentTypeError(f"No parser registered for '{source_type}'")
        return parser


def build_default_registry() -> ParserRegistry:
    from app.ingestion.parsers.docx_parser import DocxParser
    from app.ingestion.parsers.html_parser import HtmlParser
    from app.ingestion.parsers.markdown_parser import MarkdownParser
    from app.ingestion.parsers.pdf_parser import PdfParser
    from app.ingestion.parsers.txt_parser import TxtParser

    registry = ParserRegistry()
    for parser in (PdfParser(), DocxParser(), MarkdownParser(), TxtParser(), HtmlParser()):
        registry.register(parser)
    return registry
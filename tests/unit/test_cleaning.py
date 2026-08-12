"""Unit tests for deterministic text cleaning."""

from __future__ import annotations

from app.ingestion.cleaning import clean_text


def test_collapses_whitespace_and_trims() -> None:
    assert clean_text("  hello    world  ") == "hello world"


def test_removes_control_and_zero_width_chars() -> None:
    assert clean_text("a\x00b\u200bc\x7fd") == "abcd"


def test_repairs_line_break_hyphenation() -> None:
    assert clean_text("exam-\nple") == "example"


def test_keeps_genuine_hyphens() -> None:
    assert clean_text("state-of-the-art") == "state-of-the-art"


def test_collapses_excessive_newlines() -> None:
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_unicode_normalization() -> None:
    # NFKC turns the ligature ﬁ into fi
    assert clean_text("ﬁle") == "file"
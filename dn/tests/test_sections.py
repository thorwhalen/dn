"""Tests for dn.sections: cutting PDFs and markdown into addressable sections."""

import io

import pytest

from dn.sections import (
    FRONT_MATTER_TITLE,
    OutlineEntry,
    markdown_sections,
    pdf_outline,
    pdf_sections,
    slugify,
)

pypdf = pytest.importorskip("pypdf")


def _make_pdf(pages, outline):
    """A PDF whose pages hold the given lines of text, with the given bookmarks.

    ``pages``: list of lists of lines. ``outline``: list of
    ``(title, page_index, parent_title_or_None)``.
    """
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
        for line in lines:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            ops += [f"({escaped}) Tj", "T*"]
        ops.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
    refs = {}
    for title, page_index, parent in outline:
        refs[title] = writer.add_outline_item(
            title, page_index, parent=refs.get(parent)
        )
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def book_pdf():
    pages = [
        ["A Small Book", "by Nobody"],  # 1: front matter, before any bookmark
        ["Chapter 1: Beginnings", "Opening words.", "First Steps", "Step text."],  # 2
        ["More step text.", "Second Steps", "Other text."],  # 3
        ["Endings", "Closing words."],  # 4
    ]
    outline = [
        ("Chapter 1: Beginnings", 1, None),
        ("First Steps", 1, "Chapter 1: Beginnings"),
        ("Second Steps", 2, "Chapter 1: Beginnings"),
        ("Chapter 2: Endings", 3, None),
    ]
    return _make_pdf(pages, outline)


def test_pdf_outline_levels_and_pages(book_pdf):
    assert pdf_outline(book_pdf) == [
        OutlineEntry(1, "Chapter 1: Beginnings", 2, "2"),
        OutlineEntry(2, "First Steps", 2, "2"),
        OutlineEntry(2, "Second Steps", 3, "3"),
        OutlineEntry(1, "Chapter 2: Endings", 4, "4"),
    ]
    assert [e.title for e in pdf_outline(book_pdf, max_level=1)] == [
        "Chapter 1: Beginnings",
        "Chapter 2: Endings",
    ]


def test_pdf_sections_cut_inside_pages(book_pdf):
    sections = pdf_sections(book_pdf)
    by_title = {s.title: s for s in sections}

    assert [s.title for s in sections] == [
        FRONT_MATTER_TITLE,
        "Chapter 1: Beginnings",
        "First Steps",
        "Second Steps",
        "Chapter 2: Endings",
    ]
    assert [s.seq for s in sections] == list(range(5))
    # A chapter's own text stops where its first sub-section starts.
    chapter = by_title["Chapter 1: Beginnings"]
    assert "Opening words." in chapter.text and "Step text." not in chapter.text
    # A section that runs onto the next page carries a page marker and its range.
    first = by_title["First Steps"]
    assert "Step text." in first.text and "More step text." in first.text
    assert "<!-- page 3 -->" in first.text and "Other text." not in first.text
    assert (first.page_start, first.page_end) == (2, 3)
    assert by_title["Second Steps"].path == ("Chapter 1: Beginnings", "Second Steps")
    # A title printed without its numbering prefix is still found.
    assert by_title["Chapter 2: Endings"].text.startswith("Endings")


def test_pdf_sections_partition_the_text(book_pdf):
    reader = pypdf.PdfReader(io.BytesIO(book_pdf))
    whole = "".join(page.extract_text() for page in reader.pages)
    joined = "".join(s.text for s in pdf_sections(book_pdf, page_marker=None))
    assert "".join(joined.split()) == "".join(whole.split())


def test_pdf_without_outline_says_what_to_do():
    with pytest.raises(ValueError, match="markdown_sections"):
        pdf_sections(_make_pdf([["Just text."]], []))


def test_pdf_sections_page_text_seam(book_pdf):
    sections = pdf_sections(
        book_pdf, page_text=lambda page: page.extract_text().upper()
    )
    by_title = {s.title: s for s in sections}
    # The swapped extractor's text is what gets cut (titles still found, case aside).
    assert by_title["First Steps"].text.startswith("FIRST STEPS")
    assert "OPENING WORDS." in by_title["Chapter 1: Beginnings"].text


def test_markdown_sections_levels_fences_and_lines():
    md = "\n".join(
        [
            "<!-- Source: somewhere -->",  # 1: comment-only preamble is dropped
            "# Title",  # 2
            "intro",  # 3
            "## One",  # 4
            "```",  # 5
            "# not a heading",  # 6
            "```",  # 7
            "#### too deep",  # 8
            "## Two ##",  # 9
            "end",  # 10
        ]
    )
    sections = markdown_sections(md, first_seq=10, parents=("Book",))

    assert [(s.title, s.level, s.line_start, s.line_end) for s in sections] == [
        ("Title", 1, 2, 3),
        ("One", 2, 4, 8),
        ("Two", 2, 9, 10),
    ]
    assert sections[0].seq == 10 and sections[2].key == "0012-two"
    assert sections[1].path == ("Book", "Title", "One")
    assert "# not a heading" in sections[1].text
    assert "\n".join(s.text for s in sections) == "\n".join(md.splitlines()[1:])


def test_markdown_without_headings_is_one_preamble():
    (only,) = markdown_sections("just a paragraph")
    assert only.title == "(preamble)" and only.text == "just a paragraph"
    assert markdown_sections("") == []


def test_slugify():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("x" * 100, max_length=10) == "x" * 10

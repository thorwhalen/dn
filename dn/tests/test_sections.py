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
    for n, (title, page_index, parent) in enumerate(outline):
        # Keyed by position too, so two bookmarks may share a title.
        refs[title] = refs[n] = writer.add_outline_item(
            title, page_index, parent=refs.get(parent)
        )
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# The end of a section spilling onto the next page: longer than a running header.
SPILL = "More step text, a full printed line that runs well past what a running header would hold."


@pytest.fixture
def book_pdf():
    pages = [
        ["A Small Book", "by Nobody"],  # 1: front matter, before any bookmark
        ["Chapter 1: Beginnings", "Opening words.", "First Steps", "Step text."],  # 2
        [SPILL, "Second Steps", "Other text."],  # 3
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
    chapter = "Chapter 1: Beginnings"
    assert pdf_outline(book_pdf) == [
        OutlineEntry(1, chapter, 2, "2", (chapter,)),
        OutlineEntry(2, "First Steps", 2, "2", (chapter, "First Steps")),
        OutlineEntry(2, "Second Steps", 3, "3", (chapter, "Second Steps")),
        OutlineEntry(1, "Chapter 2: Endings", 4, "4", ("Chapter 2: Endings",)),
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
    assert "Step text." in first.text and SPILL in first.text
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


def test_markdown_edge_cases():
    # A longer fence is not closed by a shorter one, nor by the other character.
    md = "# A\n````\n```\n# inside\n~~~\n````\n# B"
    assert [s.title for s in markdown_sections(md)] == ["A", "B"]
    # Closing hashes need a space before them; indented headings count; CRLF is fine.
    assert [s.title for s in markdown_sections("# C#\r\n   ## Two ##\r\n")] == [
        "C#",
        "Two",
    ]
    assert markdown_sections("# A\n", parents="Book")[0].path == ("Book", "A")
    assert markdown_sections("# A\nx\n")[0].line_end == 2


def test_markdown_without_headings_is_one_preamble():
    (only,) = markdown_sections("just a paragraph")
    assert only.title == "(preamble)" and only.text == "just a paragraph"
    assert markdown_sections("") == []


def test_slugify():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("x" * 100, max_length=10) == "x" * 10


def test_same_page_parent_and_child_are_both_kept():
    # A chapter and its first section share a page and even a title: the chapter
    # keeps its heading and intro, the section starts at the *second* occurrence.
    pdf = _make_pdf(
        [["Introduction", "chapter intro", "Introduction", "sub body"]],
        [("Introduction", 0, None), ("Introduction", 0, 0)],
    )
    chapter, section = pdf_sections(pdf)
    assert (chapter.seq, section.seq) == (1, 2)
    assert chapter.text.split() == ["Introduction", "chapter", "intro"]
    assert section.text.split() == ["Introduction", "sub", "body"]
    assert section.path == ("Introduction", "Introduction")


def test_unfindable_title_starts_after_the_previous_title():
    pdf = _make_pdf(
        [["Testing with pytest", "body one", "body two"]],
        [("Testing with pytest", 0, None), ("An Image Heading", 0, 0)],
    )
    chapter, section = pdf_sections(pdf)
    assert chapter.text.strip() == "Testing with pytest"
    assert section.text.split() == ["body", "one", "body", "two"]


def test_seq_does_not_depend_on_front_matter():
    pages = [["My Book", "Alpha", "a"], ["My Book", "Beta", "b"]]
    pdf = _make_pdf(pages, [("Alpha", 0, None), ("Beta", 1, None)])
    # "My Book" is a short running header: it stays with its section, so there
    # is no front matter; the entries are numbered from 1 all the same.
    assert [(s.seq, s.title) for s in pdf_sections(pdf)] == [(1, "Alpha"), (2, "Beta")]
    assert pdf_sections(pdf)[1].text.split() == ["My", "Book", "Beta", "b"]
    # With header_chars=0 the same header becomes front matter, and seqs hold.
    assert [(s.seq, s.title) for s in pdf_sections(pdf, header_chars=0)] == [
        (0, FRONT_MATTER_TITLE),
        (1, "Alpha"),
        (2, "Beta"),
    ]


def test_chapter_label_above_the_title_stays_with_the_chapter():
    pdf = _make_pdf(
        [["Chapter 1: Start", "one"], ["CHAPTER 2", "Pythonic Code", "two"]],
        [("Chapter 1: Start", 0, None), ("Chapter 2: Pythonic Code", 1, None)],
    )
    first, second = pdf_sections(pdf)
    assert (first.page_start, first.page_end) == (
        1,
        1,
    ) and "CHAPTER 2" not in first.text
    assert second.text.split()[:2] == ["CHAPTER", "2"]


def test_title_matching_respects_word_and_line_boundaries():
    from dn.sections import _find_title

    assert _find_title(
        "Summary of the previous chapter.\nSummary\nbody", "Summary"
    ) == (33, 40)
    assert _find_title("Part II was covered.\nPart I\n", "Part I") == (21, 27)
    assert _find_title("tail about the Database design.", "Data") is None
    # Only real numbering is stripped: not a first word, not a year-like title.
    assert _find_title("The Protocol\n", "HTTP - The Protocol") is None
    assert _find_title("Pythonic Code\n", "3.1 - Pythonic Code") == (0, 13)


def test_grouping_bookmark_without_destination_takes_its_childs_page():
    pdf = _make_pdf(
        [["Ch1", "x"], ["Sec A", "y"]],
        [("Ch1", 0, None), ("Part Two", None, None), ("Sec A", 1, 1)],
    )
    entries = pdf_outline(pdf)
    assert [(e.title, e.page, e.path) for e in entries] == [
        ("Ch1", 1, ("Ch1",)),
        ("Part Two", 2, ("Part Two",)),
        ("Sec A", 2, ("Part Two", "Sec A")),
    ]


def test_outline_argument_cuts_a_pdf_that_has_no_bookmarks():
    pdf = _make_pdf([["One", "a"], ["Two", "b"]], [])
    sections = pdf_sections(
        pdf, outline=[OutlineEntry(1, "One", 1), OutlineEntry(1, "Two", 2)]
    )
    assert [(s.seq, s.title, s.page_start, s.label_start) for s in sections] == [
        (1, "One", 1, "1"),
        (2, "Two", 2, "2"),
    ]
    assert sections[0].path == ("One",)


def test_bad_page_marker_fails_before_any_work(book_pdf):
    with pytest.raises(KeyError):
        pdf_sections(book_pdf, page_marker="{nope}")


def test_accepts_a_stream(book_pdf):
    assert len(pdf_outline(io.BytesIO(book_pdf))) == 4


# ---- setext headings and EPUB tables of contents (#9) ---------------------------------


def test_setext_headings():
    md = "Guide\n=====\nIntro.\n\nSetup\nsteps\n-----\nInstall.\n\n- item\n---\n\n```\nCode\n====\n```\n"
    got = [(s.level, s.title, s.line_start) for s in markdown_sections(md)]
    # A two-line paragraph is one title; a list item over --- is a list and a rule;
    # underlines inside fenced code are code.
    assert got == [(1, "Guide", 1), (2, "Setup steps", 5)]
    assert [s.title for s in markdown_sections(md, max_level=1)] == ["Guide"]


def test_setext_and_atx_mix_and_thematic_break():
    md = "# A\ntext\n\n---\n\nB\n===\n## C\n"
    assert [(s.level, s.title) for s in markdown_sections(md)] == [(1, "A"), (1, "B"), (2, "C")]


def _epub(tmp_path):
    epub = pytest.importorskip("ebooklib.epub")
    book = epub.EpubBook()
    book.set_identifier("dn-test")
    book.set_title("A Guide")
    book.set_language("en")
    cover = epub.EpubHtml(title="Cover", file_name="cover.xhtml", content="<h1>A Guide</h1><p>By someone.</p>")
    ch1 = epub.EpubHtml(
        title="Start",
        file_name="text/ch1.xhtml",
        content='<h1>Start</h1><p>Why.</p><h2 id="setup">Setup</h2><p>Install it.</p><h2 id="use">Use</h2><p>Run it.</p>',
    )
    ch2 = epub.EpubHtml(title="End", file_name="text/ch2.xhtml", content="<h1>End</h1><p>Bye.</p>")
    for c in (cover, ch1, ch2):
        book.add_item(c)
    book.toc = [
        (epub.Section("Start", "text/ch1.xhtml"), [epub.Link("text/ch1.xhtml#setup", "Setup", "s"), epub.Link("text/ch1.xhtml#use", "Use", "u")]),
        epub.Link("text/ch2.xhtml", "End", "e"),
        epub.Link("text/missing.xhtml", "Lost", "l"),
    ]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [cover, ch1, ch2]
    path = tmp_path / "guide.epub"
    epub.write_epub(str(path), book)
    return path


def test_ebook_sections_follow_the_table_of_contents(tmp_path):
    from dn.sections import ebook_sections

    path = _epub(tmp_path)
    sections = ebook_sections(path)
    got = [(s.seq, s.level, s.title, s.path) for s in sections]
    assert got == [
        (0, 1, "(front matter)", ("(front matter)",)),
        (1, 1, "Start", ("Start",)),
        (2, 2, "Setup", ("Start", "Setup")),
        (3, 2, "Use", ("Start", "Use")),
        (4, 1, "End", ("End",)),
        (5, 1, "Lost", ("Lost",)),
    ]
    text = {s.title: s.text for s in sections}
    assert "By someone" in text["(front matter)"]
    assert "Why." in text["Start"] and "Install" not in text["Start"]  # cut inside the chapter
    assert "Install it." in text["Setup"] and "Run it." in text["Use"]
    assert "Bye." in text["End"] and text["Lost"] == ""  # a missing target keeps its key
    assert sections[2].href == "text/ch1.xhtml#setup" and sections[2].key == "0002-setup"
    assert [s.title for s in ebook_sections(path.read_bytes(), max_level=1)] == ["(front matter)", "Start", "End", "Lost"]


def test_setext_review_cases():
    def titles(md):
        return [(s.level, s.title) for s in markdown_sections(md)]

    # YAML front matter is not a heading: it stays the preamble, as before setext.
    assert titles("---\ntitle: X\nauthor: Y\n---\n\n# Ch 1\ntext\n") == [(1, "(preamble)"), (1, "Ch 1")]
    # A rule ends a block; the paragraph after it is the heading.
    assert titles("x\n\n---\nscene two\n---\n") == [(1, "(preamble)"), (2, "scene two")]
    # A list or a quote interrupts a paragraph: the --- below is a rule.
    assert titles("Intro\n- a\n- b\n---\n") == [(1, "(preamble)")]
    assert titles("Intro\n> q\n---\n") == [(1, "(preamble)")]
    # Indented code and HTML blocks are not paragraphs.
    assert titles("text\n\n    x = 1\n---\n") == [(1, "(preamble)")]
    assert titles("<!-- c -->\n---\n") == [(1, "(preamble)")]
    # A starred rule, then a real setext heading.
    assert titles("* * *\nTitle\n===\n") == [(1, "(preamble)"), (1, "Title")]


def test_ebook_sections_review_cases(tmp_path):
    epub = pytest.importorskip("ebooklib.epub")
    from dn.sections import ebook_sections

    book = epub.EpubBook()
    book.set_identifier("dn-test-2")
    book.set_title("Edge cases")
    book.set_language("en")
    one = epub.EpubHtml(
        title="One",
        file_name="text/ch 1.xhtml",  # a space: the TOC href is percent-encoded
        content='<h1>One</h1><p>First.</p><h2><a id="s1"></a>Inside</h2><p>Anchored.</p><p><a name="old"></a>Named.</p>',
    )
    two = epub.EpubHtml(title="Two", file_name="text/ch2.xhtml", content="<h1>Two</h1><p>Second.</p>")
    book.add_item(one)
    book.add_item(two)
    book.toc = [
        epub.Link("text/ch2.xhtml", "Two", "t"),  # TOC order differs from reading order
        epub.Link("text/ch%201.xhtml", "One", "o"),
        epub.Link("text/ch%201.xhtml#s1", "Inside", "i"),
        epub.Link("text/ch%201.xhtml#old", "Named", "n"),
    ]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [one, two]
    path = tmp_path / "edge.epub"
    epub.write_epub(str(path), book)

    sections = ebook_sections(path)
    by = {s.title: s for s in sections}
    assert "(front matter)" not in by  # no XML declaration or <title> text leaks in
    assert all("xml version" not in s.text for s in sections)
    assert [s.title for s in sections] == ["One", "Inside", "Named", "Two"]  # reading order
    assert by["Two"].seq == 1 and by["One"].seq == 2  # keys follow the TOC
    assert "First." in by["One"].text and "Anchored" not in by["One"].text
    assert by["Inside"].text.startswith("## Inside")  # cut before the heading, not inside it
    assert by["Named"].text.strip().startswith("Named.") and "Named." not in by["Inside"].text

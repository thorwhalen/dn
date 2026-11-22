"""Split a document into addressable sections.

A long document (a book, a report) is easier to search, cite and read when it is
a sequence of sections, each with a stable key, a title, its place in the
hierarchy and a locator (page range for a PDF, line range for markdown), than
when it is one big string. This module produces that sequence:

- :func:`pdf_outline`: a PDF's bookmarks as :class:`OutlineEntry` records.
- :func:`pdf_sections`: the PDF's text, cut at its outline entries.
- :func:`markdown_sections`: markdown text, cut at its headings (ATX ``#`` and
  setext underlines).
- :func:`ebook_sections`: an EPUB's text, cut at its own table of contents (the
  publisher's hierarchy, whatever headings the converter emits).

All ``*_sections`` functions return the same :class:`Section` records, in
reading order, and a section's own text stops where its first sub-section
starts. Together the sections hold the whole text (give or take whitespace, and
a markdown preamble made only of HTML comments, which is dropped).

**Keys are stable** for the same input and the same ``max_level``: in a PDF,
outline entry number *k* always gets ``seq`` *k* (``seq`` 0 is reserved for the
text before the first entry, whether or not there is any), so a change of text
extractor moves boundaries but never renumbers.

For an ebook without a usable table of contents, compose with the converter:
``markdown_sections(dn.ebook_to_markdown('book.epub'))``.

>>> md = '''# Guide
... Intro.
... ## Setup
... Install it.
... ## Use
... Run it.
... '''
>>> for s in markdown_sections(md):
...     print(s.key, s.level, ' > '.join(s.path), (s.line_start, s.line_end))
0000-guide 1 Guide (1, 2)
0001-setup 2 Guide > Setup (3, 4)
0002-use 2 Guide > Use (5, 6)
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Callable, Iterable
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any, Optional, Union

DFLT_MAX_LEVEL = 3
DFLT_SLUG_MAX_LENGTH = 60
DFLT_PAGE_MARKER = "\n\n<!-- page {label} -->\n\n"
#: Text above a section's title on its page is taken as a running header or a
#: "CHAPTER 2" label, and kept with the section, when it is at most this long
#: (about one printed line). Longer text is the end of the previous section.
DFLT_HEADER_CHARS = 80
FRONT_MATTER_TITLE = "(front matter)"
PREAMBLE_TITLE = "(preamble)"

PdfSource = Union[bytes, str, Path, Any]  # Any: a binary stream or a pypdf.PdfReader

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_THEMATIC_BREAK = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
# Lines that cannot start a paragraph: indented code, HTML blocks.
_NOT_PARAGRAPH_START = re.compile(r"^(?: {4}|\t| {0,3}<)")
# A line that opens a list item, a block quote or a table row: a setext underline
# below such a paragraph is a thematic break (or a table rule), not a heading.
_NOT_SETEXT_PARAGRAPH = re.compile(r"^ {0,3}(?:[-*+>|]|\d{1,9}[.)])(?:\s|$)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# "Chapter 2: ", "3.1 - ", "IV. ", "A. ": numbering a page layout often prints
# apart from the title proper. A keyword alone, or a plain first word, is not one.
_NUMBERING_PREFIX = re.compile(
    r"^\s*(?:(?i:chapter|part|appendix|section)\s+)?"
    r"(?:\d+(?:\.\d+)*|[IVXLC]+|[A-Z])\s*[:.\-–—]\s+"
)


@dataclass(frozen=True)
class OutlineEntry:
    """One bookmark of a PDF outline.

    ``page`` is the 1-based physical page; ``label`` is the number printed on
    that page (``'xii'``, ``'37'``), which is the physical number as a string
    when the PDF declares no page labels. ``path`` holds the titles from the
    outermost ancestor bookmark down to this one.
    """

    level: int
    title: str
    page: int
    label: Optional[str] = None
    path: tuple = ()


@dataclass(frozen=True)
class Section:
    """One section of a document.

    ``seq`` is the section's position in reading order and, with the slug of
    its title, forms its ``key``. ``path`` holds the titles from the outermost
    ancestor down to the section itself. The locator fields are keyword-only:
    page fields are set for PDFs, line fields (1-based, inclusive) for markdown.
    """

    seq: int
    level: int
    title: str
    path: tuple
    text: str
    _: KW_ONLY
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    label_start: Optional[str] = None
    label_end: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    #: For an EPUB section: the table-of-contents target (``'ch02.xhtml#s3'``).
    href: Optional[str] = None

    @property
    def slug(self) -> str:
        """A filename-safe form of the title."""
        return slugify(self.title)

    @property
    def key(self) -> str:
        """A stable, sortable identifier: ``'0042-the-title'``."""
        return f"{self.seq:04d}-{self.slug}"


def slugify(title: str, *, max_length: int = DFLT_SLUG_MAX_LENGTH) -> str:
    """Reduce ``title`` to lowercase ASCII letters, digits and hyphens.

    Accents are folded; a title with nothing left gives ``'section'``.

    >>> slugify('Chapter 2: Pythonic Code!')
    'chapter-2-pythonic-code'
    >>> slugify('Café déjà vu'), slugify('***')
    ('cafe-deja-vu', 'section')
    """
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "section"


# --------------------------------------------------------------------------------------
# Markdown


def markdown_sections(
    markdown: str,
    *,
    max_level: int = DFLT_MAX_LEVEL,
    first_seq: int = 0,
    parents: Iterable[str] = (),
) -> list:
    """Cut markdown at its headings of level ``max_level`` or shallower.

    ATX headings (``# Title``) and setext headings (a paragraph underlined with
    ``===`` for level 1 or ``---`` for level 2) count; a setext heading's title
    is its whole paragraph, on one line. Deeper headings stay inside their parent
    section, and headings inside fenced code are ignored. Text before the first heading becomes a
    ``'(preamble)'`` section unless it is blank or only HTML comments.

    Args:
        markdown: The markdown text.
        max_level: Deepest heading level that starts a section.
        first_seq: ``seq`` of the first section, to number several documents
            (the chapters of a book) as one sequence.
        parents: Titles to prepend to every section's ``path``.

    Returns:
        A list of :class:`Section`, in reading order. Line numbers count
        ``\\n``-separated lines.

    >>> [s.title for s in markdown_sections('hello\\n# A\\n#### deep\\n# B')]
    ['(preamble)', 'A', 'B']
    >>> [(s.level, s.title) for s in markdown_sections('Title\\n=====\\ntext\\n\\nPart\\n----')]
    [(1, 'Title'), (2, 'Part')]
    """
    lines = markdown.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":  # the text ended with a newline: not a line
        lines.pop()
    parents = (parents,) if isinstance(parents, str) else tuple(parents)
    cuts = list(_heading_cuts(lines, max_level=max_level))  # (line index, level, title)
    sections: list = []

    def add(level, title, path, start, end):
        sections.append(
            Section(
                seq=first_seq + len(sections),
                level=level,
                title=title,
                path=parents + path,
                text="\n".join(lines[start:end]),
                line_start=start + 1,
                line_end=max(end, start + 1),
            )
        )

    first_cut = cuts[0][0] if cuts else len(lines)
    if _HTML_COMMENT.sub("", "\n".join(lines[:first_cut])).strip():
        add(1, PREAMBLE_TITLE, (PREAMBLE_TITLE,), 0, first_cut)

    ancestors: dict = {}
    for k, (start, level, title) in enumerate(cuts):
        end = cuts[k + 1][0] if k + 1 < len(cuts) else len(lines)
        ancestors = {lv: t for lv, t in ancestors.items() if lv < level}
        ancestors[level] = title
        add(level, title, tuple(ancestors[lv] for lv in sorted(ancestors)), start, end)
    return sections


def _yaml_front_matter_end(lines) -> int:
    """Index just past a YAML front-matter block at the very top (``---`` ... ``---``), else 0."""
    if not lines or lines[0].strip() != "---":
        return 0
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return i + 1
    return 0


def _heading_cuts(lines, *, max_level):
    """``(line index, level, title)`` for every ATX or setext heading."""
    fence = ""  # the open fence's marker, e.g. '```' or '~~~~'
    para_start = None  # first line of the paragraph being read, if any
    skip_until = _yaml_front_matter_end(lines)  # also: past an underline just used
    for i, line in enumerate(lines):
        if i < skip_until:
            continue
        m = _FENCE.match(line)
        if fence:
            # Only the same character, at least as long, closes a fence.
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = ""
            continue
        if m:
            fence, para_start = m.group(1), None
            continue
        h = _HEADING.match(line)
        if h:
            para_start = None
            if len(h.group(1)) <= max_level:
                yield i, len(h.group(1)), h.group(2).strip()
            continue
        if not line.strip():
            para_start = None
            continue
        if para_start is None:
            # A rule, a list item, a quote, a table row, indented code or an HTML
            # block is not a paragraph, so a line of === or --- under it is not an
            # underline.
            if (
                _THEMATIC_BREAK.match(line)
                or _SETEXT_UNDERLINE.match(line)
                or _NOT_SETEXT_PARAGRAPH.match(line)
                or _NOT_PARAGRAPH_START.match(line)
            ):
                continue
            para_start = i
        elif _NOT_SETEXT_PARAGRAPH.match(line) or _THEMATIC_BREAK.match(line):
            para_start = None  # a list item, quote or rule interrupts the paragraph
            continue
        under = _SETEXT_UNDERLINE.match(lines[i + 1]) if i + 1 < len(lines) else None
        if under:
            level = 1 if under.group(1)[0] == "=" else 2
            if level <= max_level:
                title = " ".join(x.strip() for x in lines[para_start : i + 1])
                yield para_start, level, title
            para_start, skip_until = None, i + 2


# --------------------------------------------------------------------------------------
# PDF


def pdf_outline(pdf: PdfSource, *, max_level: Optional[int] = None) -> list:
    """Read a PDF's outline (its bookmarks) as a flat list of :class:`OutlineEntry`.

    A bookmark that only groups others (no destination of its own) takes the
    page of its first descendant. A bookmark whose destination cannot be
    resolved, and that has no descendant to borrow from, is left out, with a
    warning that says how many were. A PDF without bookmarks gives an empty list.

    Args:
        pdf: PDF bytes, a path, a binary stream, or an open ``pypdf.PdfReader``.
        max_level: Keep only entries this deep (1 = chapters only).

    >>> pdf_outline('book.pdf', max_level=1)  # doctest: +SKIP
    [OutlineEntry(level=1, title='Preface', page=13, label='xi', path=('Preface',)), ...]
    """
    reader = _pdf_reader(pdf)
    labels = _page_labels(reader)
    n_pages = len(reader.pages)
    unresolved = 0

    def tree(items):
        """pypdf's ``[item, [children], item, ...]`` as ``[title, index, children]`` nodes."""
        nodes: list = []
        for item in items:
            if isinstance(item, list):
                children = tree(item)
                if nodes:
                    nodes[-1][2].extend(children)
                else:  # children with no parent bookmark before them
                    nodes.extend(children)
                continue
            try:
                index = reader.get_destination_page_number(item)
            except Exception:  # pypdf raises several types for a broken destination
                index = None
            if index is not None and not 0 <= index < n_pages:
                index = None
            nodes.append([" ".join(str(item.title or "").split()), index, []])
        return nodes

    def flatten(nodes, level, ancestors):
        nonlocal unresolved
        for title, index, children in nodes:
            path = ancestors + (title,) if title else ancestors
            below = list(flatten(children, level + 1, path))
            if index is None and below:
                index = below[0].page - 1
            if title and index is not None:
                if max_level is None or level <= max_level:
                    label = labels[index] if index < len(labels) else None
                    yield OutlineEntry(level, title, index + 1, label, path)
            elif title:
                unresolved += 1
            yield from below

    entries = list(flatten(tree(reader.outline), 1, ()))
    if unresolved:
        warnings.warn(
            f"{unresolved} bookmark(s) point at no page and were left out of the outline",
            stacklevel=2,
        )
    return entries


def pdf_sections(
    pdf: PdfSource,
    *,
    max_level: int = DFLT_MAX_LEVEL,
    outline: Optional[Iterable] = None,
    page_text: Optional[Callable] = None,
    page_marker: Optional[str] = DFLT_PAGE_MARKER,
    header_chars: int = DFLT_HEADER_CHARS,
) -> list:
    """Cut a PDF's text at its outline entries.

    A section starts where its title is found on its outline page and ends
    where the next section starts. When the title cannot be located (it is an
    image, or the extractor garbled it) the section starts at the top of its
    page, or right after the previous section's title when they share a page.
    Text before the first entry becomes a ``'(front matter)'`` section.

    Args:
        pdf: PDF bytes, a path, a binary stream, or an open ``pypdf.PdfReader``.
        max_level: Deepest outline level that starts a section.
        outline: The entries to cut at, as :class:`OutlineEntry` records, in
            place of the PDF's own bookmarks: for a PDF that has none (one
            entry per chapter you know the page of, or one per page).
        page_text: ``page -> str``, to swap the text extractor (an OCR result,
            another library). Defaults to pypdf's ``page.extract_text()``.
        page_marker: Inserted where a new page starts *inside* a section, so a
            reader can cite the page. A format string with ``{label}`` (the
            printed page number) and ``{page}`` (the 1-based physical page).
            ``None`` for no markers.
        header_chars: Text above a section's title on its page stays with that
            section when it is at most this long (a running header, a "CHAPTER
            2" label); longer text is the end of the previous section.

    Returns:
        A list of :class:`Section`, in reading order. Entry number *k* of the
        outline has ``seq`` *k* (from 1); ``seq`` 0 is the front matter.

    Raises:
        ValueError: If there is no outline to cut at; the message says what to
            pass instead.

    >>> sections = pdf_sections('book.pdf')  # doctest: +SKIP
    >>> sections[40].key, sections[40].page_start, sections[40].label_start  # doctest: +SKIP
    ('0040-the-repository-pattern', 61, '33')
    """
    reader = _pdf_reader(pdf)
    if page_marker is not None:
        page_marker.format(
            label="1", page=1
        )  # a bad placeholder fails here, not mid-run
    if outline is not None:
        entries = [e for e in outline if e.level <= max_level]
    else:
        entries = pdf_outline(reader, max_level=max_level)
    if not entries:
        raise ValueError(
            "No outline to cut at: this PDF has no bookmarks (or none at this "
            "max_level). Pass outline=[OutlineEntry(level, title, page), ...] for the "
            "headings you know, or cut dn.pdf_to_markdown's output with "
            "markdown_sections (one section per page)."
        )
    # Outline order is reading order in a well-formed PDF; a stable sort repairs
    # the rare out-of-order bookmark without disturbing same-page entries.
    entries = sorted(entries, key=lambda e: e.page)
    labels = _page_labels(reader)
    extract = page_text or (lambda page: page.extract_text())
    n_pages = len(reader.pages)
    cache: dict = {}

    def text_of(index):
        if index not in cache:
            cache[index] = extract(reader.pages[index]) or ""
        return cache[index]

    def label_of(index):
        return labels[index] if index < len(labels) else str(index + 1)

    # Where each entry starts: (page index, character offset in that page's text).
    starts: list = []
    search_from = 0  # on the current page: just past the previous entry's title
    for entry in entries:
        index = min(entry.page, n_pages) - 1
        same_page = bool(starts) and starts[-1][0] == index
        floor = search_from if same_page else 0
        found = _find_title(text_of(index), entry.title, start=floor)
        if found is None:
            start, search_from = floor, floor
        else:
            start, search_from = found
            above = text_of(index)[:start]
            if not same_page and len(above.strip()) <= header_chars:
                start = 0  # a running header or chapter label: it belongs here
        starts.append((index, start))

    def span(start, end):
        """Text from ``start`` up to (not including) ``end``, both (page, offset)."""
        (p0, o0), (p1, o1) = start, end
        if p0 == p1:
            return text_of(p0)[o0:o1], p0
        parts = [text_of(p0)[o0:]]
        last = p0
        for index in range(p0 + 1, min(p1, n_pages - 1) + 1):
            chunk = text_of(index) if index < p1 else text_of(index)[:o1]
            if not chunk.strip():
                continue
            if page_marker is not None:
                parts.append(page_marker.format(label=label_of(index), page=index + 1))
            else:
                parts.append("\n")
            parts.append(chunk)
            last = index
        return "".join(parts), last

    sections: list = []

    def add(seq, level, title, path, start, end):
        text, last = span(start, end)
        sections.append(
            Section(
                seq=seq,
                level=level,
                title=title,
                path=path,
                text=text.strip("\n"),
                page_start=start[0] + 1,
                page_end=last + 1,
                label_start=label_of(start[0]),
                label_end=label_of(last),
            )
        )

    if span((0, 0), starts[0])[0].strip():
        add(0, 1, FRONT_MATTER_TITLE, (FRONT_MATTER_TITLE,), (0, 0), starts[0])
    for k, entry in enumerate(entries):
        end = starts[k + 1] if k + 1 < len(entries) else (n_pages, 0)
        add(
            k + 1,
            entry.level,
            entry.title,
            entry.path or (entry.title,),
            starts[k],
            end,
        )
    return sections


# --------------------------------------------------------------------------------------
# EPUB

_MARK = "DNSECTIONMARK{:06d}"
_BLOCK_TAGS = (
    "p",
    "div",
    "section",
    "article",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "table",
    "figure",
)
_MARK_RE = re.compile(r"DNSECTIONMARK(\d{6})")


def _flatten_toc(toc, max_level, level=1, ancestors=()):
    """``(level, title, href, path)`` for each table-of-contents entry, in order.

    ebooklib gives ``Link`` objects and ``(Section, [children])`` tuples.
    """
    for node in toc:
        head, children = (node[0], node[1]) if isinstance(node, tuple) else (node, ())
        title = (getattr(head, "title", "") or "").strip()
        href = getattr(head, "href", "") or ""
        if level <= max_level:
            path = ancestors + (title,)
            if title:
                yield level, title, href, path
            yield from _flatten_toc(children, max_level, level + 1, path)


def ebook_sections(
    ebook: Union[bytes, str, Path, Any],
    *,
    max_level: int = DFLT_MAX_LEVEL,
    to_markdown: Optional[Callable[[str], str]] = None,
) -> list:
    """Cut an EPUB at its own table of contents.

    Each entry's section starts at its target (a document, or an element inside
    it by ``id``) and ends where the next section starts, in reading (spine)
    order, so a chapter holding several entries is cut inside. The hierarchy is
    the publisher's, whatever headings the text has. Text before the first entry
    becomes a ``'(front matter)'`` section.

    Args:
        ebook: EPUB bytes, a path, or a binary stream.
        max_level: Deepest table-of-contents level that starts a section.
        to_markdown: ``html -> markdown`` for each spine document; defaults to
            markdownify with dn's settings (as ``dn.ebook_to_markdown``'s
            ``ebooklib`` backend).

    Returns:
        A list of :class:`Section` (``href`` set). Entry number *k* of the table
        of contents has ``seq`` *k* (from 1); ``seq`` 0 is the front matter. An
        entry whose target is not in the book gets an empty section, so keys
        never shift.

    Raises:
        ValueError: If the book has no table of contents; the message says what
            to use instead.

    >>> sections = ebook_sections('book.epub')  # doctest: +SKIP
    >>> sections[3].key, sections[3].path, sections[3].href  # doctest: +SKIP
    ('0003-setup', ('Guide', 'Setup'), 'ch01.xhtml#setup')
    """
    import io
    import posixpath
    from urllib.parse import unquote

    try:
        import ebooklib
        from ebooklib import epub

        from dn.ebook import body_soup
    except ImportError as e:
        raise ImportError(
            "ebook_sections needs the 'ebook' extra: EbookLib and markdownify (which brings bs4)."
        ) from e
    if to_markdown is None:
        from markdownify import MarkdownConverter

        from dn.src import HTML_TO_MARKDOWN_DEFAULTS, _collapse_blank_runs

        converter = MarkdownConverter(**HTML_TO_MARKDOWN_DEFAULTS)

        def to_markdown(html):
            return _collapse_blank_runs(converter.convert(html))

    if isinstance(ebook, bytes):
        ebook = io.BytesIO(ebook)
    book = epub.read_epub(ebook if hasattr(ebook, "read") else str(ebook))
    entries = list(_flatten_toc(book.toc, max_level))
    if not entries:
        raise ValueError(
            "No table of contents to cut at (or none at this max_level). Cut the "
            "converted text at its headings instead: "
            "markdown_sections(dn.ebook_to_markdown(src))."
        )

    spine = [book.get_item_with_id(idref) for idref, *_ in book.spine]
    spine = [
        d for d in spine if d is not None and d.get_type() == ebooklib.ITEM_DOCUMENT
    ]

    names = [posixpath.normpath(d.get_name()) for d in spine]

    def file_of(href_file):
        """The spine document an href's file part names: exact first, then a unique
        path suffix (a TOC in a sub-folder writes paths relative to itself)."""
        target = posixpath.normpath(unquote(href_file))
        if target in names:
            return target
        hits = [
            n for n in names if n.endswith("/" + target) or target.endswith("/" + n)
        ]
        return hits[0] if len(hits) == 1 else None

    targets = []  # per entry: (spine document name or None, fragment)
    for _, _, href, _ in entries:
        file, _, fragment = href.partition("#")
        targets.append((file_of(file), unquote(fragment)))

    chunks = []
    for doc, name in zip(spine, names):
        soup = body_soup(doc.get_content())
        at_top = 0
        for k, (file, fragment) in enumerate(targets):
            if file != name:
                continue
            marker = soup.new_tag("p")
            marker.string = _MARK.format(k)
            target = None
            if fragment:
                target = soup.find(id=fragment) or soup.find(attrs={"name": fragment})
            if target is not None:
                # An anchor inside a heading (<h2><a id=..></a>Title</h2>): cut
                # before the heading, not inside it.
                if target.name not in _BLOCK_TAGS:
                    target = target.find_parent(_BLOCK_TAGS) or target
                target.insert_before(marker)
            else:  # the document itself, or an anchor that is not there
                soup.insert(at_top, marker)
                at_top += 1
        chunks.append(to_markdown(str(soup)).strip())
    text = "\n\n".join(c for c in chunks if c)

    marks = list(_MARK_RE.finditer(text))
    position = {int(m.group(1)): j for j, m in enumerate(marks)}
    spans = {}  # entry index -> its text
    for j, m in enumerate(marks):
        end = marks[j + 1].start() if j + 1 < len(marks) else len(text)
        spans[int(m.group(1))] = text[m.end() : end].strip()
    sections: list = []
    front = text[: marks[0].start()].strip() if marks else text.strip()
    if front:
        sections.append(Section(0, 1, FRONT_MATTER_TITLE, (FRONT_MATTER_TITLE,), front))
    # Reading order: where each entry's text starts; an entry with no target last.
    order = sorted(range(len(entries)), key=lambda k: (position.get(k, len(marks)), k))
    for k in order:
        level, title, href, path = entries[k]
        sections.append(Section(k + 1, level, title, path, spans.get(k, ""), href=href))
    return sections


def _pdf_reader(pdf: PdfSource):
    """Open ``pdf`` (bytes, path, stream or reader) as a ``pypdf.PdfReader``."""
    if hasattr(pdf, "pages") and hasattr(pdf, "outline"):
        return pdf
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - exercised only without pypdf
        raise ImportError(
            "Reading PDF outlines needs pypdf: pip install 'dn[pdf]'"
        ) from e
    if isinstance(pdf, (bytes, bytearray)):
        import io

        return PdfReader(io.BytesIO(bytes(pdf)))
    if hasattr(pdf, "read"):
        return PdfReader(pdf)
    return PdfReader(str(Path(pdf).expanduser()))


def _page_labels(reader) -> list:
    try:
        return list(reader.page_labels)
    except Exception:  # malformed /PageLabels: fall back to physical numbers
        return []


def _find_title(page_text: str, title: str, *, start: int = 0) -> Optional[tuple]:
    """Where ``title`` is in ``page_text`` at or after ``start``: ``(start, end)`` or ``None``.

    Whitespace is matched loosely (extracted text breaks titles across lines).
    A title alone on its line(s) is preferred, then one that starts a line and
    ends at a word boundary; each is tried case-sensitively first. The same is
    then tried without a leading numbering prefix (``'Chapter 2: '``), which
    page layouts often print apart from the title proper. A title that only
    occurs mid-sentence is not a heading, and is not matched.

    >>> _find_title('intro text\\nPythonic\\nCode\\nbody', 'Chapter 2: Pythonic Code')
    (11, 24)
    >>> _find_title('Part II was covered earlier.\\nPart I\\nbody', 'Part I')
    (29, 35)
    >>> _find_title('a tail about the Database design.', 'Data') is None
    True
    """
    candidates = [title]
    stripped = _NUMBERING_PREFIX.sub("", title, count=1)
    if stripped and stripped != title:
        candidates.append(stripped)
    for candidate in candidates:
        words = candidate.split()
        if not words:
            continue
        body = r"\s+".join(re.escape(w) for w in words)
        whole_line = r"^[ \t]*(" + body + r")[ \t]*$"
        line_start = r"^[ \t]*(" + body + r")(?!\w)"
        for pattern, flags in (
            (whole_line, re.MULTILINE),
            (whole_line, re.MULTILINE | re.IGNORECASE),
            (line_start, re.MULTILINE),
            (line_start, re.MULTILINE | re.IGNORECASE),
        ):
            m = re.compile(pattern, flags).search(page_text, start)
            if m:
                return m.start(1), m.end(1)
    return None

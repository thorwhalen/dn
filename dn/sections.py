"""Split a document into addressable sections.

A long document (a book, a report) is easier to search, cite and read when it is
a sequence of sections, each with a stable key, a title, its place in the
hierarchy and a locator (page range for a PDF, line range for markdown), than
when it is one big string. This module produces that sequence:

- :func:`pdf_outline`: a PDF's bookmarks as ``(level, title, page, label)``.
- :func:`pdf_sections`: the PDF's text, cut at its outline entries.
- :func:`markdown_sections`: markdown text, cut at its headings.

Both ``*_sections`` functions return the same :class:`Section` records, and both
*partition* the text: every character belongs to exactly one section, and a
section's own text stops where its first sub-section starts.

For an ebook, compose with the converter:
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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

DFLT_MAX_LEVEL = 3
DFLT_SLUG_MAX_LENGTH = 60
DFLT_PAGE_MARKER = "\n\n<!-- page {label} -->\n\n"
FRONT_MATTER_TITLE = "(front matter)"
PREAMBLE_TITLE = "(preamble)"

PdfSource = Union[bytes, str, Path, Any]  # Any: an already-open pypdf.PdfReader

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_NUMBERING_PREFIX = re.compile(
    r"^\s*(?:(?:chapter|part|appendix|section)\s+)?[\dA-Z]+(?:\.\d+)*\s*[:.\-–—]\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OutlineEntry:
    """One bookmark of a PDF outline.

    ``page`` is the 1-based physical page; ``label`` is the page number printed
    on it (``'xii'``, ``'37'``) when the PDF declares page labels.
    """

    level: int
    title: str
    page: int
    label: Optional[str] = None


@dataclass(frozen=True)
class Section:
    """One section of a document.

    ``seq`` is the section's position in reading order and, with the slug of
    its title, forms its ``key``. ``path`` holds the titles from the outermost
    ancestor down to the section itself. Page fields are set for PDFs, line
    fields (1-based, inclusive) for markdown.
    """

    seq: int
    level: int
    title: str
    path: tuple
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    label_start: Optional[str] = None
    label_end: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    @property
    def slug(self) -> str:
        """A filename-safe form of the title."""
        return slugify(self.title)

    @property
    def key(self) -> str:
        """A stable, sortable identifier: ``'0042-the-title'``."""
        return f"{self.seq:04d}-{self.slug}"


def slugify(title: str, *, max_length: int = DFLT_SLUG_MAX_LENGTH) -> str:
    """Lowercase ``title`` and reduce it to letters, digits and hyphens.

    >>> slugify('Chapter 2: Pythonic Code!')
    'chapter-2-pythonic-code'
    >>> slugify('***')
    'section'
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
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

    Deeper headings stay inside their parent section, and headings inside code
    fences are ignored. Text before the first heading becomes a
    ``'(preamble)'`` section unless it is blank or only HTML comments.

    Args:
        markdown: The markdown text.
        max_level: Deepest heading level that starts a section.
        first_seq: ``seq`` of the first section, to number several documents
            (the chapters of a book) as one sequence.
        parents: Titles to prepend to every section's ``path``.

    Returns:
        A list of :class:`Section`, in reading order.

    >>> [s.title for s in markdown_sections('hello\\n# A\\n#### deep\\n# B')]
    ['(preamble)', 'A', 'B']
    """
    lines = markdown.splitlines()
    parents = tuple(parents)
    cuts = list(_heading_cuts(lines, max_level=max_level))  # (line index, level, title)
    sections: list = []

    def add(level, title, path, start, end):
        text = "\n".join(lines[start:end])
        sections.append(
            Section(
                seq=first_seq + len(sections),
                level=level,
                title=title,
                path=parents + path,
                text=text,
                line_start=start + 1,
                line_end=max(end, start + 1),
            )
        )

    first_cut = cuts[0][0] if cuts else len(lines)
    preamble = "\n".join(lines[:first_cut])
    if _HTML_COMMENT.sub("", preamble).strip():
        add(1, PREAMBLE_TITLE, (PREAMBLE_TITLE,), 0, first_cut)

    ancestors: dict = {}
    for k, (start, level, title) in enumerate(cuts):
        end = cuts[k + 1][0] if k + 1 < len(cuts) else len(lines)
        ancestors = {lv: t for lv, t in ancestors.items() if lv < level}
        ancestors[level] = title
        add(level, title, tuple(ancestors[lv] for lv in sorted(ancestors)), start, end)
    return sections


def _heading_cuts(lines, *, max_level):
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING.match(line)
        if m and len(m.group(1)) <= max_level:
            yield i, len(m.group(1)), m.group(2).strip()


# --------------------------------------------------------------------------------------
# PDF


def pdf_outline(pdf: PdfSource, *, max_level: Optional[int] = None) -> list:
    """Read a PDF's outline (its bookmarks) as a flat list of :class:`OutlineEntry`.

    Entries whose destination cannot be resolved to a page are skipped. A PDF
    without bookmarks gives an empty list.

    Args:
        pdf: PDF bytes, a path, or an open ``pypdf.PdfReader``.
        max_level: Keep only entries this deep (1 = chapters only).

    >>> pdf_outline('book.pdf', max_level=1)  # doctest: +SKIP
    [OutlineEntry(level=1, title='Preface', page=13, label='xi'), ...]
    """
    reader = _pdf_reader(pdf)
    labels = _page_labels(reader)

    def walk(items, level):
        for item in items:
            if isinstance(item, list):
                yield from walk(item, level + 1)
                continue
            if max_level is not None and level > max_level:
                continue
            try:
                index = reader.get_destination_page_number(item)
            except Exception:  # a broken destination: skip the bookmark
                continue
            if index is None or index < 0:
                continue
            title = " ".join(str(item.title or "").split())
            if not title:
                continue
            label = labels[index] if index < len(labels) else None
            yield OutlineEntry(level=level, title=title, page=index + 1, label=label)

    return list(walk(reader.outline, 1))


def pdf_sections(
    pdf: PdfSource,
    *,
    max_level: int = DFLT_MAX_LEVEL,
    page_text: Optional[Callable] = None,
    page_marker: Optional[str] = DFLT_PAGE_MARKER,
) -> list:
    """Cut a PDF's text at its outline entries.

    A section starts where its title is found on its outline page (at the top
    of that page when the title cannot be located) and ends where the next
    section starts. Text before the first entry becomes a ``'(front matter)'``
    section.

    Args:
        pdf: PDF bytes, a path, or an open ``pypdf.PdfReader``.
        max_level: Deepest outline level that starts a section.
        page_text: ``page -> str``, to swap the text extractor. Defaults to
            pypdf's ``page.extract_text()``.
        page_marker: Inserted where a new page starts *inside* a section, so a
            reader can cite the page. A format string with ``{label}`` (the
            printed page number, or the physical one) and ``{page}`` (the
            1-based physical page). ``None`` for no markers.

    Returns:
        A list of :class:`Section`, in reading order.

    Raises:
        ValueError: If the PDF has no outline. Convert it with
            ``dn.pdf_to_markdown`` and cut that with :func:`markdown_sections`,
            or pass the headings yourself.

    >>> sections = pdf_sections('book.pdf')  # doctest: +SKIP
    >>> sections[40].key, sections[40].page_start, sections[40].label_start  # doctest: +SKIP
    ('0040-the-repository-pattern', 61, '33')
    """
    reader = _pdf_reader(pdf)
    entries = pdf_outline(reader, max_level=max_level)
    if not entries:
        raise ValueError(
            "This PDF has no outline (bookmarks) to cut at. Convert it with "
            "dn.pdf_to_markdown and use markdown_sections on the result."
        )
    # Outline order is reading order in a well-formed PDF; a stable sort repairs
    # the rare out-of-order bookmark without disturbing same-page entries.
    entries = sorted(entries, key=lambda e: e.page)
    labels = _page_labels(reader)
    extract = page_text or (lambda page: page.extract_text() or "")
    n_pages = len(reader.pages)
    cache: dict = {}

    def text_of(index):
        if index not in cache:
            cache[index] = extract(reader.pages[index])
        return cache[index]

    def label_of(index):
        return labels[index] if index < len(labels) else None

    # Where each entry starts: (page index, character offset in that page's text).
    starts = []
    for entry in entries:
        index = entry.page - 1
        floor = starts[-1][1] if starts and starts[-1][0] == index else 0
        offset = _find_title(text_of(index), entry.title, start=floor)
        starts.append((index, floor if offset is None else offset))

    def span(start, end):
        """Text from ``start`` up to (not including) ``end``, both (page, offset)."""
        (p0, o0), (p1, o1) = start, end
        if p0 == p1:
            return text_of(p0)[o0:o1]
        parts = [text_of(p0)[o0:]]
        for index in range(p0 + 1, min(p1, n_pages - 1) + 1):
            chunk = text_of(index) if index < p1 else text_of(index)[:o1]
            if index == p1 and not chunk.strip():
                continue
            if page_marker is not None:
                parts.append(
                    page_marker.format(
                        label=label_of(index) or index + 1, page=index + 1
                    )
                )
            else:
                parts.append("\n")
            parts.append(chunk)
        return "".join(parts)

    end_of_document = (n_pages, 0)
    sections: list = []

    def add(level, title, path, start, end):
        last = end[0] if end[1] > 0 else end[0] - 1
        last = max(start[0], min(last, n_pages - 1))
        sections.append(
            Section(
                seq=len(sections),
                level=level,
                title=title,
                path=path,
                text=span(start, end).strip("\n"),
                page_start=start[0] + 1,
                page_end=last + 1,
                label_start=label_of(start[0]),
                label_end=label_of(last),
            )
        )

    if span((0, 0), starts[0]).strip():
        add(1, FRONT_MATTER_TITLE, (FRONT_MATTER_TITLE,), (0, 0), starts[0])

    ancestors: dict = {}
    for k, entry in enumerate(entries):
        end = starts[k + 1] if k + 1 < len(entries) else end_of_document
        ancestors = {lv: t for lv, t in ancestors.items() if lv < entry.level}
        ancestors[entry.level] = entry.title
        path = tuple(ancestors[lv] for lv in sorted(ancestors))
        add(entry.level, entry.title, path, starts[k], end)
    return sections


def _pdf_reader(pdf: PdfSource):
    """Open ``pdf`` (bytes, path or reader) as a ``pypdf.PdfReader``."""
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
    return PdfReader(str(Path(pdf).expanduser()))


def _page_labels(reader) -> list:
    try:
        return list(reader.page_labels)
    except Exception:  # malformed /PageLabels: do without printed numbers
        return []


def _find_title(page_text: str, title: str, *, start: int = 0) -> Optional[int]:
    """Offset of ``title`` in ``page_text`` at or after ``start``, else ``None``.

    Whitespace is matched loosely (extracted text breaks titles across lines).
    A line-start, case-sensitive match is preferred; then the same without a
    leading numbering prefix (``'Chapter 2: '``), which page layouts often set
    apart from the title proper.

    >>> _find_title('intro text\\nPythonic\\nCode\\nbody', 'Chapter 2: Pythonic Code')
    11
    >>> _find_title('nothing here', 'Missing') is None
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
        for pattern, flags in (
            (r"^[ \t]*" + body, re.MULTILINE),
            (r"^[ \t]*" + body, re.MULTILINE | re.IGNORECASE),
            (body, 0),
        ):
            m = re.compile(pattern, flags).search(page_text, start)
            if m:
                return m.start()
    return None

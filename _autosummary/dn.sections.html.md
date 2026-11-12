# dn.sections

Split a document into addressable sections.

A long document (a book, a report) is easier to search, cite and read when it is
a sequence of sections, each with a stable key, a title, its place in the
hierarchy and a locator (page range for a PDF, line range for markdown), than
when it is one big string. This module produces that sequence:

- [`pdf_outline()`](#dn.sections.pdf_outline): a PDF’s bookmarks as [`OutlineEntry`](#dn.sections.OutlineEntry) records.
- [`pdf_sections()`](#dn.sections.pdf_sections): the PDF’s text, cut at its outline entries.
- [`markdown_sections()`](#dn.sections.markdown_sections): markdown text, cut at its ATX (`#`) headings.

Both `*_sections` functions return the same [`Section`](#dn.sections.Section) records, in
reading order, and a section’s own text stops where its first sub-section
starts. Together the sections hold the whole text (give or take whitespace, and
a markdown preamble made only of HTML comments, which is dropped).

**Keys are stable** for the same input and the same `max_level`: in a PDF,
outline entry number *k* always gets `seq` *k* (`seq` 0 is reserved for the
text before the first entry, whether or not there is any), so a change of text
extractor moves boundaries but never renumbers.

For an ebook, compose with the converter:
`markdown_sections(dn.ebook_to_markdown('book.epub'))`.

```pycon
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
```

### Module Attributes

| [`DFLT_HEADER_CHARS`](#dn.sections.DFLT_HEADER_CHARS)   | Text above a section's title on its page is taken as a running header or a "CHAPTER 2" label, and kept with the section, when it is at most this long (about one printed line).   |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Functions

| [`markdown_sections`](#dn.sections.markdown_sections)(markdown, \*[, max_level, ...])   | Cut markdown at its headings of level `max_level` or shallower.                                                       |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [`pdf_outline`](#dn.sections.pdf_outline)(pdf, \*[, max_level])                   | Read a PDF's outline (its bookmarks) as a flat list of [`OutlineEntry`](#dn.sections.OutlineEntry). |
| [`pdf_sections`](#dn.sections.pdf_sections)(pdf, \*[, max_level, outline, ...])    | Cut a PDF's text at its outline entries.                                                                              |
| [`slugify`](#dn.sections.slugify)(title, \*[, max_length])                    | Reduce `title` to lowercase ASCII letters, digits and hyphens.                                                        |

### Classes

| [`OutlineEntry`](#dn.sections.OutlineEntry)(level, title, page[, label, path])   | One bookmark of a PDF outline.   |
|----------------------------------------------------------------------------------------------------|----------------------------------|
| [`Section`](#dn.sections.Section)(seq, level, title, path, text, \*[, ...]) | One section of a document.       |

### dn.sections.DFLT_HEADER_CHARS *= 80*

Text above a section’s title on its page is taken as a running header or a
“CHAPTER 2” label, and kept with the section, when it is at most this long
(about one printed line). Longer text is the end of the previous section.

### *class* dn.sections.OutlineEntry(level, title, page, label=None, path=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One bookmark of a PDF outline.

`page` is the 1-based physical page; `label` is the number printed on
that page (`'xii'`, `'37'`), which is the physical number as a string
when the PDF declares no page labels. `path` holds the titles from the
outermost ancestor bookmark down to this one.

### *class* dn.sections.Section(seq, level, title, path, text, , page_start=None, page_end=None, label_start=None, label_end=None, line_start=None, line_end=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One section of a document.

`seq` is the section’s position in reading order and, with the slug of
its title, forms its `key`. `path` holds the titles from the outermost
ancestor down to the section itself. The locator fields are keyword-only:
page fields are set for PDFs, line fields (1-based, inclusive) for markdown.

#### *property* key *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

`'0042-the-title'`.

* **Type:**
  A stable, sortable identifier

#### *property* slug *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

A filename-safe form of the title.

### dn.sections.markdown_sections(markdown, , max_level=3, first_seq=0, parents=())

Cut markdown at its headings of level `max_level` or shallower.

Only ATX headings (`# Title`) count; underlined (setext) headings are not
recognised. Deeper headings stay inside their parent section, and headings
inside fenced code are ignored. Text before the first heading becomes a
`'(preamble)'` section unless it is blank or only HTML comments.

* **Parameters:**
  * **markdown** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The markdown text.
  * **max_level** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Deepest heading level that starts a section.
  * **first_seq** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – `seq` of the first section, to number several documents
    (the chapters of a book) as one sequence.
  * **parents** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Titles to prepend to every section’s `path`.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)
* **Returns:**
  A list of [`Section`](#dn.sections.Section), in reading order. Line numbers count
  `\n`-separated lines.

```pycon
>>> [s.title for s in markdown_sections('hello\n# A\n#### deep\n# B')]
['(preamble)', 'A', 'B']
```

### dn.sections.pdf_outline(pdf, , max_level=None)

Read a PDF’s outline (its bookmarks) as a flat list of [`OutlineEntry`](#dn.sections.OutlineEntry).

A bookmark that only groups others (no destination of its own) takes the
page of its first descendant. A bookmark whose destination cannot be
resolved, and that has no descendant to borrow from, is left out, with a
warning that says how many were. A PDF without bookmarks gives an empty list.

* **Parameters:**
  * **pdf** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – PDF bytes, a path, a binary stream, or an open `pypdf.PdfReader`.
  * **max_level** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Keep only entries this deep (1 = chapters only).
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

```pycon
>>> pdf_outline('book.pdf', max_level=1)
[OutlineEntry(level=1, title='Preface', page=13, label='xi', path=('Preface',)), ...]
```

### dn.sections.pdf_sections(pdf, , max_level=3, outline=None, page_text=None, page_marker='\\n\\n<!-- page {label} -->\\n\\n', header_chars=80)

Cut a PDF’s text at its outline entries.

A section starts where its title is found on its outline page and ends
where the next section starts. When the title cannot be located (it is an
image, or the extractor garbled it) the section starts at the top of its
page, or right after the previous section’s title when they share a page.
Text before the first entry becomes a `'(front matter)'` section.

* **Parameters:**
  * **pdf** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – PDF bytes, a path, a binary stream, or an open `pypdf.PdfReader`.
  * **max_level** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Deepest outline level that starts a section.
  * **outline** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)]) – The entries to cut at, as [`OutlineEntry`](#dn.sections.OutlineEntry) records, in
    place of the PDF’s own bookmarks: for a PDF that has none (one
    entry per chapter you know the page of, or one per page).
  * **page_text** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]) – `page -> str`, to swap the text extractor (an OCR result,
    another library). Defaults to pypdf’s `page.extract_text()`.
  * **page_marker** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Inserted where a new page starts *inside* a section, so a
    reader can cite the page. A format string with `{label}` (the
    printed page number) and `{page}` (the 1-based physical page).
    `None` for no markers.
  * **header_chars** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Text above a section’s title on its page stays with that
    section when it is at most this long (a running header, a “CHAPTER
    2” label); longer text is the end of the previous section.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)
* **Returns:**
  A list of [`Section`](#dn.sections.Section), in reading order. Entry number *k* of the
  outline has `seq` *k* (from 1); `seq` 0 is the front matter.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If there is no outline to cut at; the message says what to
      pass instead.

```pycon
>>> sections = pdf_sections('book.pdf')
>>> sections[40].key, sections[40].page_start, sections[40].label_start
('0040-the-repository-pattern', 61, '33')
```

### dn.sections.slugify(title, , max_length=60)

Reduce `title` to lowercase ASCII letters, digits and hyphens.

Accents are folded; a title with nothing left gives `'section'`.

```pycon
>>> slugify('Chapter 2: Pythonic Code!')
'chapter-2-pythonic-code'
>>> slugify('Café déjà vu'), slugify('***')
('cafe-deja-vu', 'section')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

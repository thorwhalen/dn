# dn - Document-to-Markdown Conversion

dn converts documents (PDF, Word, Excel, PowerPoint, HTML, Jupyter notebooks) to
markdown, and provides markdown repair/cleanup utilities.

## Package Structure

```
dn/
  src.py     # Format converters: bytes_to_markdown, pdf_to_markdown, etc.
  ebook.py   # Ebook (EPUB/MOBI/AZW3/...) to markdown, via a backend registry
  ocr.py     # OCR for scanned (text-layer-less) PDFs
  to.py      # Reverse: markdown_to_notebook
  repair.py  # Markdown repair: repair_markdown, remove_hyperlink_crap, etc.
  util.py    # Utilities: path handling, URL fetching
```

## Key Functions

- `bytes_to_markdown(data, input_format=None, key=None)` — smart converter with format detection
- `repair_markdown(md)` — fix broken links, empty links, and other artifacts
- `remove_hyperlink_crap(text)` — strip ChatGPT/Claude citation artifacts
- `remove_improperly_double_newlines(text)` — fix spurious whitespace-only blank lines
- `notebook_to_markdown(nb_source)` — Jupyter notebook to markdown
- `ebook_to_markdown(src)` — ebook (path, URL or bytes) to markdown
- `check_ebook_requirements()` — which ebook backends work here, and how to fix the rest
- `ocr_pdf_to_markdown(pdf_bytes)` — OCR a scanned PDF; `pdf_to_markdown` calls it automatically

## Dependencies

Core: `dol`. Optional extras: `dn[pdf]`, `dn[word]`, `dn[html]`, `dn[ebook]`, `dn[ocr]`, `dn[all]`.

Ebook conversion prefers *system* tools over pip packages: calibre's
`ebook-convert` plus `pandoc` give by far the best markdown, because calibre
resolves the publisher's CSS into real character formatting before pandoc
renders it. `dn/ebook.py` degrades to pandoc-only, calibre-only, then pure-python
(`ebooklib`) when those are missing.

`dn/ocr.py` exists because `pdf_to_markdown` on a scanned book used to emit page
headers and no text — an empty result that looked like success. `ocr='auto'`
(the default) engages only when the *whole* document is text-less, so a text PDF
with figure pages never pays the ~1s/page cost; `ocr=True` OCRs every text-less
page. `'auto'` swallows OCR failures (a missing language pack must not sink a
directory-wide conversion); `ocr=True` raises.

Two format sets in `dn/ebook.py`, deliberately: `EBOOK_FORMATS` is what
`ebook_to_markdown` will attempt on request, `AUTOWIRED_EBOOK_FORMATS` is the
narrower set wired into `dflt_converters` and filename detection. Extensions
calibre reads but that usually mean something else (`.rb`, `.pdb`, `.opf`,
`.prc`, `.textile`) are in the first and not the second — auto-claiming them
routed ordinary source files through calibre.

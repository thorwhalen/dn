# dn.ocr

Read text out of PDFs that have no text layer, by OCR.

A scanned book is a PDF full of *pictures* of pages. `pypdf` and friends
extract nothing from it – not an error, just an empty string – so
`dn.src.pdf_to_markdown()` silently produces a markdown file with headers and
no content. The fix is optical character recognition: render each page to an
image and run it through Tesseract.

The everyday use is not to call this module at all. `pdf_to_markdown` consults
it automatically for pages that yield no text, so scanned PDFs simply work:

```pycon
>>> from dn import pdf_to_markdown
>>> md = pdf_to_markdown(scanned_pdf_bytes)  # OCRs the empty pages
```

Call it directly when you want control over resolution, language, or which pages
to process:

```pycon
>>> md = ocr_pdf_to_markdown(pdf_bytes, dpi=400, lang='eng+fra')
```

OCR is slow – expect roughly a second per page – and needs two things that are
not pure-python dependencies: the Tesseract binary, and PyMuPDF to rasterize
pages. [`check_ocr_requirements()`](#dn.ocr.check_ocr_requirements) reports what’s present and how to install
what isn’t.

```pycon
>>> report = check_ocr_requirements(verbose=False)
>>> sorted(report)
['pymupdf', 'pytesseract', 'tesseract']
```

### Module Attributes

| [`DFLT_OCR_DPI`](#dn.ocr.DFLT_OCR_DPI)   | Rendering resolution.       |
|-----------------------------------------------------------------|-----------------------------|
| [`DFLT_OCR_LANG`](#dn.ocr.DFLT_OCR_LANG)  | Tesseract language pack(s). |

### Functions

| [`ocr_pdf_to_markdown`](#dn.ocr.ocr_pdf_to_markdown)(pdf_bytes, \*[, dpi, ...])   | OCR an entire PDF and return it as markdown, one section per page.   |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`ocr_pdf_pages`](#dn.ocr.ocr_pdf_pages)(pdf_bytes, \*[, pages, dpi, ...])  | OCR selected pages of a PDF.                                         |
| [`check_ocr_requirements`](#dn.ocr.check_ocr_requirements)(\*[, verbose])            | Report what OCR needs, what's present, and how to install the rest.  |
| [`find_tesseract`](#dn.ocr.find_tesseract)()                                 | Find the `tesseract` binary, or `None` if it isn't installed.        |
| [`ocr_is_available`](#dn.ocr.ocr_is_available)()                               | Whether everything needed to OCR a PDF is present.                   |

### Exceptions

| [`OcrError`](#dn.ocr.OcrError)   | Raised when OCR could not be performed.   |
|-------------------------------------------------------------|-------------------------------------------|

### dn.ocr.DFLT_OCR_DPI *= 300*

Rendering resolution. 300 DPI is the usual floor for reliable OCR of body
text; below it Tesseract starts dropping small type and footnotes.

### dn.ocr.DFLT_OCR_LANG *= 'eng'*

Tesseract language pack(s). Combine with ‘+’, e.g. `'eng+fra'`.

### *exception* dn.ocr.OcrError

Bases: [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

Raised when OCR could not be performed.

### dn.ocr.check_ocr_requirements(, verbose=True)

Report what OCR needs, what’s present, and how to install the rest.

* **Parameters:**
  **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print a human-readable report as well as returning it.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  `{requirement: {'available': bool, 'install': str | None}}`.

### Example

```pycon
>>> report = check_ocr_requirements(verbose=False)
>>> set(report['tesseract']) == {'available', 'install'}
True
```

### dn.ocr.find_tesseract()

Find the `tesseract` binary, or `None` if it isn’t installed.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> find_tesseract()
'/opt/homebrew/bin/tesseract'
```

### dn.ocr.ocr_is_available()

Whether everything needed to OCR a PDF is present.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> isinstance(ocr_is_available(), bool)
True
```

### dn.ocr.ocr_pdf_pages(pdf_bytes, , pages=None, dpi=300, lang='eng', max_workers=None)

OCR selected pages of a PDF.

Pages are rendered in small batches and OCR’d concurrently. Tesseract runs as
a subprocess, so threads genuinely parallelize it, and batching keeps the
rendered images from piling up in memory on a long book.

* **Parameters:**
  * **pdf_bytes** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – The PDF file’s bytes.
  * **pages** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]]) – Zero-based page numbers to OCR. `None` means every page.
  * **dpi** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Rendering resolution. Higher is slower and usually more accurate.
  * **lang** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Tesseract language pack(s), e.g. `'eng'` or `'eng+fra'`.
  * **max_workers** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Concurrent OCR workers. Defaults to the CPU count, capped
    at 8.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  `{page_number: text}` for the requested pages.
* **Raises:**
  [**OcrError**](#dn.ocr.OcrError) – If Tesseract, pytesseract, or PyMuPDF is missing, if
      `max_workers` or `dpi` is not positive, or if a page number is
      out of range.

### dn.ocr.ocr_pdf_to_markdown(pdf_bytes, , dpi=300, lang='eng', md_inner_file_header='###', max_workers=None)

OCR an entire PDF and return it as markdown, one section per page.

* **Parameters:**
  * **pdf_bytes** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – The PDF file’s bytes.
  * **dpi** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Rendering resolution.
  * **lang** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Tesseract language pack(s).
  * **md_inner_file_header** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Header level used for the per-page headings.
  * **max_workers** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Concurrent OCR workers.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Markdown text.
* **Raises:**
  [**OcrError**](#dn.ocr.OcrError) – If the OCR stack is not installed.

### Example

```pycon
>>> md = ocr_pdf_to_markdown(scanned_pdf_bytes)
'### Page 1\n\nTHE VISUAL DISPLAY...'
```

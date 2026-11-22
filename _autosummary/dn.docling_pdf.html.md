# dn.docling_pdf

An opt-in PDF converter backed by Docling: real tables and section headings.

`dn`’s default PDF converter (pypdf, with OCR for scanned books) is fast and light,
and it returns flat text under `### Page N` headers: a table becomes one line of
numbers, a section heading an ordinary line. [Docling](https://github.com/docling-project/docling)
(MIT code, models under CDLA-Permissive-2.0, hosted by the LF AI & Data Foundation)
recovers the layout: markdown tables and headings. It is also 50 to 200 times slower,
installs torch and downloads its models on first use, so it is never a default. Weighed
in addaix/contaix#24.

Use it for one call, through the existing `converters=` seam:

```default
from dn.src import bytes_to_markdown
from dn.docling_pdf import docling_converters
md = bytes_to_markdown(pdf_bytes, "pdf", converters=docling_converters())
```

or make it this process’s default PDF converter:

```default
from dn.src import add_dflt_converter
from dn.docling_pdf import docling_pdf_to_markdown
add_dflt_converter("pdf", docling_pdf_to_markdown)
```

`pip install 'dn[docling]'` installs it. Importing this module costs nothing: Docling
is imported on the first conversion.

```pycon
>>> docling_converters(base={"docx": str})["pdf"] is docling_pdf_to_markdown
True
```

### Module Attributes

| [`INSTALL_HINT`](#dn.docling_pdf.INSTALL_HINT)   | How to get the extra (named in the error when Docling is missing).   |
|-----------------------------------------------------------------|----------------------------------------------------------------------|

### Functions

| [`docling_converters`](#dn.docling_pdf.docling_converters)([base])                    | `base` (default: `dn.src.dflt_converters`) with Docling for `"pdf"`.            |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`docling_is_available`](#dn.docling_pdf.docling_is_available)()                        | True when the `docling` package can be imported (nothing is imported to check). |
| [`docling_pdf_to_markdown`](#dn.docling_pdf.docling_pdf_to_markdown)(pdf_bytes, \*[, ...]) | Convert PDF bytes to markdown with Docling, keeping tables and headings.        |

### dn.docling_pdf.INSTALL_HINT *= "pip install 'dn[docling]'"*

How to get the extra (named in the error when Docling is missing).

### dn.docling_pdf.docling_converters(base=None)

`base` (default: `dn.src.dflt_converters`) with Docling for `"pdf"`.

A copy: the process’s defaults are left alone.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### dn.docling_pdf.docling_is_available()

True when the `docling` package can be imported (nothing is imported to check).

```pycon
>>> isinstance(docling_is_available(), bool)
True
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### dn.docling_pdf.docling_pdf_to_markdown(pdf_bytes, , document_converter=None, name='document.pdf')

Convert PDF bytes to markdown with Docling, keeping tables and headings.

* **Parameters:**
  * **pdf_bytes** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – The PDF file’s bytes.
  * **document_converter** – A configured `docling.document_converter.DocumentConverter`
    (OCR, table mode, accelerator…). Default: one built on first use and kept
    for the process, since building one loads the models.
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The file name Docling is told; only its extension matters.
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – Docling is not installed (the message says how to install it).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

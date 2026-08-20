"""
Tools for markdown parsing and generation.
"""

from dol import Files  # just to have it ready in the namespace

from dn.to import markdown_to_notebook
from dn.repair import (
    repair_markdown,
    fix_multiline_links,
    fix_empty_links,
    remove_hyperlink_crap,
    remove_improperly_double_newlines,
    strip_repeated_lines,
)

from dn.src import (
    notebook_to_markdown,
    bytes_to_markdown,
    bytes_store_to_markdown_store,
    add_dflt_converter,
    dflt_converters,
)

from dn.ebook import (
    ebook_to_markdown,  # Convert an ebook (EPUB, MOBI, AZW3, ...) to markdown
    EBOOK_FORMATS,  # Ebook formats dn offers markdown conversion for
    EbookConversionError,  # Raised when an ebook could not be converted
    check_ebook_requirements,  # Report available backends and how to install the rest
    ebook_backends,  # Registered ebook backends, in preference order
    available_ebook_backends,  # ...restricted to those whose requirements are met
    register_ebook_backend,  # Add your own ebook-to-markdown strategy
)

from dn.ocr import (
    ocr_pdf_to_markdown,  # OCR a scanned (text-layer-less) PDF into markdown
    ocr_pdf_pages,  # OCR selected pages of a PDF
    ocr_is_available,  # Whether the OCR stack is installed
    check_ocr_requirements,  # Report OCR requirements and how to install them
    OcrError,  # Raised when OCR could not be performed
)


# --------------------------------------------------------------------------------------
from contextlib import suppress as _suppress

_ignore_import_errors = _suppress(ImportError)

with _ignore_import_errors:
    from dn.src import pdf_to_markdown  # requires pypdf

with _ignore_import_errors:
    from dn.src import docx_to_markdown  # requires mammoth

with _ignore_import_errors:
    from dn.src import excel_to_markdown  # requires pandas and openpyxl and tabulate

with _ignore_import_errors:
    from dn.src import pptx_to_markdown  # requires python-pptx

with _ignore_import_errors:
    from dn.src import html_to_markdown  # requires html2text

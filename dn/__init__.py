"""
Tools for markdown parsing and generation.
"""

from dol import Files  # just to have it ready in the namespace

from dn.to import markdown_to_notebook

from dn.src import (
    notebook_to_markdown,
    bytes_to_markdown,
    bytes_store_to_markdown_store,
    add_dflt_converter,
    dflt_converters,
)


# --------------------------------------------------------------------------------------
from contextlib import suppress as _suppress

_ignore_import_errors = _suppress(ImportError)

with _ignore_import_errors:
    from dn.src import pdf_to_markdown  # requires pypdf

with _ignore_import_errors:
    from dn.src import docx_to_markdown  # requires mammoth

with _ignore_import_errors:
    from dn.src import excel_to_markdown  # requires pandas and openpyxl

with _ignore_import_errors:
    from dn.src import pptx_to_markdown  # requires python-pptx

with _ignore_import_errors:
    from dn.src import html_to_markdown  # requires html2text

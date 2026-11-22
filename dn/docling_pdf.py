"""An opt-in PDF converter backed by Docling: real tables and section headings.

``dn``'s default PDF converter (pypdf, with OCR for scanned books) is fast and light,
and it returns flat text under ``### Page N`` headers: a table becomes one line of
numbers, a section heading an ordinary line. `Docling <https://github.com/docling-project/docling>`_
(MIT code, models under CDLA-Permissive-2.0, hosted by the LF AI & Data Foundation)
recovers the layout: markdown tables and headings. It is also 50 to 200 times slower,
installs torch and downloads its models on first use, so it is never a default. Weighed
in addaix/contaix#24.

Use it for one call, through the existing ``converters=`` seam::

    from dn.src import bytes_to_markdown
    from dn.docling_pdf import docling_converters
    md = bytes_to_markdown(pdf_bytes, "pdf", converters=docling_converters())

or make it this process's default PDF converter::

    from dn.src import add_dflt_converter
    from dn.docling_pdf import docling_pdf_to_markdown
    add_dflt_converter("pdf", docling_pdf_to_markdown)

``pip install 'dn[docling]'`` installs it. Importing this module costs nothing: Docling
is imported on the first conversion.

>>> docling_converters(base={"docx": str})["pdf"] is docling_pdf_to_markdown
True
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from functools import lru_cache
from importlib.util import find_spec

#: How to get the extra (named in the error when Docling is missing).
INSTALL_HINT = "pip install 'dn[docling]'"


def docling_is_available() -> bool:
    """True when the ``docling`` package can be imported (nothing is imported to check).

    >>> isinstance(docling_is_available(), bool)
    True
    """
    return find_spec("docling") is not None


@lru_cache(maxsize=1)
def _default_document_converter():
    """One Docling ``DocumentConverter`` per process: building it loads the models."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as error:
        raise ImportError(
            f"Docling is not installed; {INSTALL_HINT} (it pulls in torch)"
        ) from error
    return DocumentConverter()


def docling_pdf_to_markdown(
    pdf_bytes: bytes,
    *,
    document_converter=None,
    name: str = "document.pdf",
) -> str:
    """Convert PDF bytes to markdown with Docling, keeping tables and headings.

    Args:
        pdf_bytes: The PDF file's bytes.
        document_converter: A configured ``docling.document_converter.DocumentConverter``
            (OCR, table mode, accelerator...). Default: one built on first use and kept
            for the process, since building one loads the models.
        name: The file name Docling is told; only its extension matters.

    Raises:
        ImportError: Docling is not installed (the message says how to install it).
    """
    converter = document_converter or _default_document_converter()
    from docling.datamodel.base_models import DocumentStream

    source = DocumentStream(name=name, stream=io.BytesIO(pdf_bytes))
    return converter.convert(source).document.export_to_markdown()


def docling_converters(
    base: Mapping[str, Callable[[bytes], str]] | None = None,
) -> dict[str, Callable[[bytes], str]]:
    """``base`` (default: ``dn.src.dflt_converters``) with Docling for ``"pdf"``.

    A copy: the process's defaults are left alone.
    """
    if base is None:
        from dn.src import dflt_converters as base
    return {**base, "pdf": docling_pdf_to_markdown}

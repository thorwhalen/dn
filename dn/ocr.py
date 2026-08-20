"""
Read text out of PDFs that have no text layer, by OCR.

A scanned book is a PDF full of *pictures* of pages. ``pypdf`` and friends
extract nothing from it -- not an error, just an empty string -- so
:func:`dn.src.pdf_to_markdown` silently produces a markdown file with headers and
no content. The fix is optical character recognition: render each page to an
image and run it through Tesseract.

The everyday use is not to call this module at all. ``pdf_to_markdown`` consults
it automatically for pages that yield no text, so scanned PDFs simply work:

>>> from dn import pdf_to_markdown  # doctest: +SKIP
>>> md = pdf_to_markdown(scanned_pdf_bytes)  # OCRs the empty pages  # doctest: +SKIP

Call it directly when you want control over resolution, language, or which pages
to process:

>>> md = ocr_pdf_to_markdown(pdf_bytes, dpi=400, lang='eng+fra')  # doctest: +SKIP

OCR is slow -- expect roughly a second per page -- and needs two things that are
not pure-python dependencies: the Tesseract binary, and PyMuPDF to rasterize
pages. :func:`check_ocr_requirements` reports what's present and how to install
what isn't.

>>> report = check_ocr_requirements(verbose=False)
>>> sorted(report)
['pymupdf', 'pytesseract', 'tesseract']
"""

import io
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Union
from collections.abc import Callable, Iterable, Iterator, Sequence

__all__ = [
    "OcrError",
    "ocr_pdf_to_markdown",
    "ocr_pdf_pages",
    "check_ocr_requirements",
    "find_tesseract",
    "ocr_is_available",
    "DFLT_OCR_DPI",
    "DFLT_OCR_LANG",
]


#: Rendering resolution. 300 DPI is the usual floor for reliable OCR of body
#: text; below it Tesseract starts dropping small type and footnotes.
DFLT_OCR_DPI = 300

#: Tesseract language pack(s). Combine with '+', e.g. ``'eng+fra'``.
DFLT_OCR_LANG = "eng"

#: A page yielding fewer characters than this is treated as having no text layer.
#: Scanned pages typically extract 0 characters; a handful can leak in from
#: stamped page numbers or an embedded logo, hence a small non-zero threshold.
DFLT_MIN_CHARS_PER_PAGE = 20


class OcrError(RuntimeError):
    """Raised when OCR could not be performed."""


_TESSERACT_CANDIDATES = (
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/usr/bin/tesseract",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

_INSTALL_HINTS = {
    "tesseract": (
        "Tesseract OCR engine -- https://tesseract-ocr.github.io/tessdoc/Installation.html\n"
        "      macOS: brew install tesseract  |  "
        "Linux: sudo apt install tesseract-ocr  |  "
        "Windows: winget install UB-Mannheim.TesseractOCR"
    ),
    "pytesseract": "pip install 'dn[ocr]'  (installs pytesseract and PyMuPDF)",
    "pymupdf": "pip install 'dn[ocr]'  (installs pytesseract and PyMuPDF)",
}


def find_tesseract() -> str | None:
    """Find the ``tesseract`` binary, or ``None`` if it isn't installed.

    >>> find_tesseract()  # doctest: +SKIP
    '/opt/homebrew/bin/tesseract'
    """
    path = shutil.which("tesseract")
    if path:
        return path
    for candidate in _TESSERACT_CANDIDATES:
        candidate = os.path.expandvars(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _importable(module_name: str) -> bool:
    """Whether ``module_name`` can be imported, without importing it."""
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def ocr_is_available() -> bool:
    """Whether everything needed to OCR a PDF is present.

    >>> isinstance(ocr_is_available(), bool)
    True
    """
    return (
        find_tesseract() is not None
        and _importable("pytesseract")
        and _importable("fitz")  # PyMuPDF
    )


def check_ocr_requirements(*, verbose: bool = True) -> dict:
    """Report what OCR needs, what's present, and how to install the rest.

    Args:
        verbose: Print a human-readable report as well as returning it.

    Returns:
        ``{requirement: {'available': bool, 'install': str | None}}``.

    Example:

    >>> report = check_ocr_requirements(verbose=False)
    >>> set(report['tesseract']) == {'available', 'install'}
    True
    """
    present = {
        "tesseract": find_tesseract() is not None,
        "pytesseract": _importable("pytesseract"),
        "pymupdf": _importable("fitz"),
    }
    report = {
        name: {
            "available": ok,
            "install": None if ok else _INSTALL_HINTS[name],
        }
        for name, ok in present.items()
    }

    if verbose:
        print("OCR requirements:\n")
        for name, info in report.items():
            mark = "OK     " if info["available"] else "MISSING"
            print(f"  [{mark}] {name}")
            if info["install"]:
                print(f"      {info['install']}")
        if not all(info["available"] for info in report.values()):
            print("\nOCR is unavailable: scanned PDFs will yield no text.")

    return report


def _require_ocr():
    """Import the OCR stack, or raise an error naming what to install."""
    missing = [
        name
        for name, info in check_ocr_requirements(verbose=False).items()
        if not info["available"]
    ]
    if missing:
        hints = "\n".join(f"  - {_INSTALL_HINTS[name]}" for name in missing)
        raise OcrError(f"OCR needs {', '.join(missing)}, which is missing.\n{hints}")

    import fitz  # PyMuPDF
    import pytesseract

    tesseract_cmd = find_tesseract()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return fitz, pytesseract


def _render_page(doc, page_number: int, *, dpi: int) -> bytes:
    """Render one page of an open PyMuPDF document to PNG bytes."""
    pixmap = doc.load_page(page_number).get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")


def _image_to_text(image_bytes: bytes, *, pytesseract, lang: str) -> str:
    """OCR a single PNG image."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        return pytesseract.image_to_string(image, lang=lang)


def _chunked(seq: Sequence, size: int) -> Iterator[Sequence]:
    """Split a sequence into consecutive chunks of at most ``size``.

    >>> list(_chunked([1, 2, 3, 4, 5], 2))
    [[1, 2], [3, 4], [5]]
    """
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def ocr_pdf_pages(
    pdf_bytes: bytes,
    *,
    pages: Optional[Iterable[int]] = None,
    dpi: int = DFLT_OCR_DPI,
    lang: str = DFLT_OCR_LANG,
    max_workers: Optional[int] = None,
) -> dict:
    """OCR selected pages of a PDF.

    Pages are rendered in small batches and OCR'd concurrently. Tesseract runs as
    a subprocess, so threads genuinely parallelize it, and batching keeps the
    rendered images from piling up in memory on a long book.

    Args:
        pdf_bytes: The PDF file's bytes.
        pages: Zero-based page numbers to OCR. ``None`` means every page.
        dpi: Rendering resolution. Higher is slower and usually more accurate.
        lang: Tesseract language pack(s), e.g. ``'eng'`` or ``'eng+fra'``.
        max_workers: Concurrent OCR workers. Defaults to the CPU count, capped
            at 8.

    Returns:
        ``{page_number: text}`` for the requested pages.

    Raises:
        OcrError: If Tesseract, pytesseract, or PyMuPDF is missing.
    """
    fitz, pytesseract = _require_ocr()

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)

    ocr_one = partial(_image_to_text, pytesseract=pytesseract, lang=lang)
    texts = {}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_numbers = list(range(doc.page_count) if pages is None else pages)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch in _chunked(page_numbers, max_workers * 2):
                # Render on this thread (PyMuPDF documents aren't thread-safe),
                # then fan the slow part out.
                images = [_render_page(doc, n, dpi=dpi) for n in batch]
                for page_number, text in zip(batch, executor.map(ocr_one, images)):
                    texts[page_number] = text.strip()

    return texts


def ocr_pdf_to_markdown(
    pdf_bytes: bytes,
    *,
    dpi: int = DFLT_OCR_DPI,
    lang: str = DFLT_OCR_LANG,
    md_inner_file_header: str = "###",
    max_workers: Optional[int] = None,
) -> str:
    """OCR an entire PDF and return it as markdown, one section per page.

    Args:
        pdf_bytes: The PDF file's bytes.
        dpi: Rendering resolution.
        lang: Tesseract language pack(s).
        md_inner_file_header: Header level used for the per-page headings.
        max_workers: Concurrent OCR workers.

    Returns:
        Markdown text.

    Raises:
        OcrError: If the OCR stack is not installed.

    Example:

    >>> md = ocr_pdf_to_markdown(scanned_pdf_bytes)  # doctest: +SKIP
    '### Page 1\\n\\nTHE VISUAL DISPLAY...'
    """
    texts = ocr_pdf_pages(pdf_bytes, dpi=dpi, lang=lang, max_workers=max_workers)
    return "\n\n".join(
        f"{md_inner_file_header} Page {page_number + 1}\n\n{texts[page_number]}"
        for page_number in sorted(texts)
    )

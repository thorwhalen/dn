"""
Tests for OCR of PDFs that carry no text layer.

The fixture builds a genuine image-only PDF -- text drawn into a bitmap, saved as
a PDF page -- which is exactly what a scanned book is, so these exercise the real
pipeline without committing a binary.
"""

import io

import pytest

from dn.ocr import (
    DFLT_MIN_CHARS_PER_PAGE,
    OcrError,
    check_ocr_requirements,
    find_tesseract,
    ocr_is_available,
    ocr_pdf_pages,
    ocr_pdf_to_markdown,
    _chunked,
)


_SCANNED_LINES = ("Scanned pages need optical", "character recognition.")


def _scanned_pdf_bytes(lines=_SCANNED_LINES) -> bytes:
    """Build a PDF whose only content is a picture of some text."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("L", (1700, 200 + 140 * len(lines)), color=255)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:  # older Pillow: load_default takes no size
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((60, 60 + 140 * i), line, fill=0, font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="PDF", resolution=200)
    return buffer.getvalue()


@pytest.fixture
def scanned_pdf():
    return _scanned_pdf_bytes()


needs_ocr = pytest.mark.skipif(
    not ocr_is_available(),
    reason="OCR stack unavailable (needs tesseract, pytesseract, PyMuPDF)",
)


# --------------------------------------------------------------------------------------
# Availability reporting (runs anywhere)


def test_check_ocr_requirements_covers_the_whole_stack():
    report = check_ocr_requirements(verbose=False)
    assert set(report) == {"tesseract", "pytesseract", "pymupdf"}
    for info in report.values():
        # Anything missing must tell the user how to install it.
        assert info["available"] or info["install"]


def test_ocr_is_available_agrees_with_the_report():
    report = check_ocr_requirements(verbose=False)
    assert ocr_is_available() == all(i["available"] for i in report.values())


def test_find_tesseract_returns_a_path_or_none():
    result = find_tesseract()
    assert result is None or isinstance(result, str)


def test_chunked():
    assert list(_chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(_chunked([], 3)) == []


# --------------------------------------------------------------------------------------
# The premise: a scanned PDF really does defeat plain text extraction


def test_the_fixture_has_no_text_layer(scanned_pdf):
    """If this ever fails, the fixture stopped being a realistic scan."""
    pypdf = pytest.importorskip("pypdf")

    reader = pypdf.PdfReader(io.BytesIO(scanned_pdf))
    extracted = "".join(page.extract_text() or "" for page in reader.pages)
    assert len(extracted.strip()) < DFLT_MIN_CHARS_PER_PAGE


# --------------------------------------------------------------------------------------
# Real OCR (skipped when the stack isn't installed)


@needs_ocr
def test_ocr_pdf_to_markdown_recovers_the_text(scanned_pdf):
    """User story: I have a scanned book and want its words as markdown."""
    md = ocr_pdf_to_markdown(scanned_pdf)
    assert "### Page 1" in md
    for line in _SCANNED_LINES:
        assert line in md


@needs_ocr
def test_ocr_pdf_pages_selects_pages(scanned_pdf):
    texts = ocr_pdf_pages(scanned_pdf, pages=[0])
    assert set(texts) == {0}
    assert _SCANNED_LINES[0] in texts[0]


@needs_ocr
def test_pdf_to_markdown_ocrs_automatically(scanned_pdf):
    """The point of the wiring: callers get text without asking for OCR."""
    from dn import pdf_to_markdown

    md = pdf_to_markdown(scanned_pdf)
    assert _SCANNED_LINES[0] in md


@needs_ocr
def test_pdf_to_markdown_ocr_can_be_turned_off(scanned_pdf):
    from dn import pdf_to_markdown

    md = pdf_to_markdown(scanned_pdf, ocr=False)
    assert _SCANNED_LINES[0] not in md
    assert "### Page 1" in md  # structure still there, just no text


def test_pdf_to_markdown_ocr_false_never_needs_the_ocr_stack(scanned_pdf):
    """ocr=False must work on a machine with no tesseract at all."""
    from dn import pdf_to_markdown

    assert "### Page 1" in pdf_to_markdown(scanned_pdf, ocr=False)


def test_ocr_required_but_missing_raises_informatively(scanned_pdf, monkeypatch):
    """ocr=True means "you must", so an incomplete stack is an error, not a shrug."""
    monkeypatch.setattr("dn.ocr.find_tesseract", lambda: None)
    from dn import pdf_to_markdown

    with pytest.raises(OcrError, match="tesseract"):
        pdf_to_markdown(scanned_pdf, ocr=True)


def test_text_pdfs_are_not_ocred(monkeypatch):
    """A PDF with a real text layer must never pay the OCR cost."""
    from dn.tests.utils_for_testing_dn import test_data_dir
    from dol import Files
    from dn import pdf_to_markdown

    def boom(*a, **kw):
        raise AssertionError("OCR ran on a PDF that already had text")

    monkeypatch.setattr("dn.src.ocr_pdf_pages", boom)
    md = pdf_to_markdown(Files(test_data_dir)["test.pdf"])
    assert "Page 1" in md

"""The opt-in Docling PDF converter (addaix/contaix#24), with Docling stood in for."""

import sys
import types

import pytest

import dn.docling_pdf as dp


@pytest.fixture
def fake_docling(monkeypatch):
    """A ``docling`` package whose converter records what it was given."""
    seen = {}

    class DocumentStream:
        def __init__(self, *, name, stream):
            self.name, self.stream = name, stream

    class DocumentConverter:
        def convert(self, source):
            seen["name"], seen["bytes"] = source.name, source.stream.read()
            doc = types.SimpleNamespace(export_to_markdown=lambda: "| a | b |\n|---|---|")
            return types.SimpleNamespace(document=doc)

    pkg = types.ModuleType("docling")
    dc = types.ModuleType("docling.document_converter")
    dc.DocumentConverter = DocumentConverter
    dm = types.ModuleType("docling.datamodel")
    bm = types.ModuleType("docling.datamodel.base_models")
    bm.DocumentStream = DocumentStream
    for name, mod in {
        "docling": pkg,
        "docling.document_converter": dc,
        "docling.datamodel": dm,
        "docling.datamodel.base_models": bm,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    dp._default_document_converter.cache_clear()
    yield seen
    dp._default_document_converter.cache_clear()


def test_pdf_bytes_go_to_docling_and_its_markdown_comes_back(fake_docling):
    assert dp.docling_pdf_to_markdown(b"%PDF-1.7 ...") == "| a | b |\n|---|---|"
    assert fake_docling == {"name": "document.pdf", "bytes": b"%PDF-1.7 ..."}


def test_the_converter_is_built_once_per_process(fake_docling):
    dp.docling_pdf_to_markdown(b"x")
    first = dp._default_document_converter()
    dp.docling_pdf_to_markdown(b"y")
    assert dp._default_document_converter() is first


def test_it_plugs_into_bytes_to_markdown_through_converters(fake_docling):
    from dn.src import bytes_to_markdown, dflt_converters

    before = dict(dflt_converters)
    md = bytes_to_markdown(b"%PDF", "pdf", converters=dp.docling_converters())
    assert md.startswith("| a | b |")
    assert dflt_converters == before  # the process defaults are left alone


def test_without_docling_the_error_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    dp._default_document_converter.cache_clear()
    with pytest.raises(ImportError, match=r"dn\[docling\]"):
        dp.docling_pdf_to_markdown(b"%PDF")
    dp._default_document_converter.cache_clear()


def test_importing_dn_does_not_import_docling():
    import subprocess

    code = "import dn, dn.src, dn.docling_pdf, sys; print('docling' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr

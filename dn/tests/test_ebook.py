"""
Tests for ebook (EPUB, MOBI, AZW3, ...) to markdown conversion.

Split in two: registry mechanics, which run anywhere, and real conversions,
which are skipped unless a backend's external tools (calibre, pandoc) are
actually installed on the machine running the tests.
"""

import zipfile
from io import BytesIO

import pytest

from dn.ebook import (
    AUTOWIRED_EBOOK_FORMATS,
    EBOOK_FORMATS,
    ebook_formats,
    _no_backend_message,
    EbookBackend,
    EbookConversionError,
    available_ebook_backends,
    check_ebook_requirements,
    ebook_backends,
    ebook_to_markdown,
    register_ebook_backend,
    register_ebook_converters,
    sniff_ebook_format,
    _ebook_backends,
)


# --------------------------------------------------------------------------------------
# Fixtures


_CHAPTER_TEXT = "Bugs, like models, are opinions embedded in software."
_CHAPTER_TITLE = "The Only Chapter"


def _minimal_epub_bytes() -> bytes:
    """Build the smallest EPUB real converters will accept.

    Assembling it here (rather than committing a binary fixture) keeps the repo
    text-only and makes what is being converted visible to the test reader.
    """
    container = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>A Tiny Book</dc:title>
    <dc:creator>Test Author</dc:creator>
    <dc:language>en</dc:language>
    <dc:identifier id="id">urn:uuid:dn-test-book</dc:identifier>
  </metadata>
  <manifest>
    <item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
  </manifest>
  <spine toc="ncx"><itemref idref="ch1"/></spine>
</package>"""

    ncx = (
        """<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="urn:uuid:dn-test-book"/></head>
  <docTitle><text>A Tiny Book</text></docTitle>
  <navMap><navPoint id="n1" playOrder="1">
    <navLabel><text>%s</text></navLabel>
    <content src="chapter1.xhtml"/>
  </navPoint></navMap>
</ncx>"""
        % _CHAPTER_TITLE
    )

    chapter = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>%s</title></head>
<body><h1>%s</h1><p>%s</p></body></html>""" % (
        _CHAPTER_TITLE,
        _CHAPTER_TITLE,
        _CHAPTER_TEXT,
    )

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # The mimetype entry must come first and be stored uncompressed.
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("content.opf", opf)
        zf.writestr("toc.ncx", ncx)
        zf.writestr("chapter1.xhtml", chapter)
    return buffer.getvalue()


@pytest.fixture
def epub_bytes():
    return _minimal_epub_bytes()


@pytest.fixture
def epub_file(tmp_path, epub_bytes):
    path = tmp_path / "tiny_book.epub"
    path.write_bytes(epub_bytes)
    return path


# --------------------------------------------------------------------------------------
# Registry mechanics (no external tools needed)


def test_shipped_backends_are_registered_in_preference_order():
    assert ebook_backends()[0] == "calibre_pandoc"
    assert set(ebook_backends()) >= {
        "calibre_pandoc",
        "pandoc",
        "calibre_txt",
        "ebooklib",
    }


def test_backend_format_filtering():
    """ebooklib is EPUB-only, so it must not be offered for MOBI."""
    assert "ebooklib" in ebook_backends("epub")
    assert "ebooklib" not in ebook_backends("mobi")
    # A backend declaring no formats covers every format dn claims support for.
    assert "calibre_pandoc" in ebook_backends("mobi")


def test_backend_handles_respects_ebook_formats():
    backend = EbookBackend("x", lambda src, fmt: "", lambda: True)
    assert backend.handles("mobi")
    assert not backend.handles("cbz")  # image container, deliberately excluded


def test_register_and_use_a_custom_backend(epub_bytes):
    """User story: I have my own converter and want dn to prefer it."""
    register_ebook_backend(
        "shouty",
        lambda src, fmt: "HELLO FROM SHOUTY",
        is_available=lambda: True,
        formats=["epub"],
        priority=0,  # ahead of everything shipped
    )
    try:
        assert ebook_backends("epub")[0] == "shouty"
        assert ebook_to_markdown(epub_bytes) == "HELLO FROM SHOUTY"
    finally:
        del _ebook_backends["shouty"]


def test_registering_an_existing_name_needs_force():
    with pytest.raises(ValueError, match="already registered"):
        register_ebook_backend("pandoc", lambda src, fmt: "", is_available=lambda: True)


def test_unavailable_backends_are_skipped(epub_bytes):
    """A backend that says it isn't available must never be called."""
    register_ebook_backend(
        "broken",
        lambda src, fmt: pytest.fail("unavailable backend was called"),
        is_available=lambda: False,
        formats=["epub"],
        priority=0,  # first in line, so only availability can stop it
    )
    try:
        assert "broken" not in available_ebook_backends("epub")
        if available_ebook_backends("epub"):
            # The assertion that matters: dispatching must skip it entirely.
            assert _CHAPTER_TEXT in ebook_to_markdown(epub_bytes)
    finally:
        del _ebook_backends["broken"]


def test_failed_backend_falls_through_to_the_next(epub_bytes):
    """User story: my preferred converter chokes; dn should try the next one."""
    register_ebook_backend(
        "explodes",
        lambda src, fmt: (_ for _ in ()).throw(RuntimeError("kaboom")),
        is_available=lambda: True,
        formats=["epub"],
        priority=0,
    )
    register_ebook_backend(
        "works",
        lambda src, fmt: "RECOVERED",
        is_available=lambda: True,
        formats=["epub"],
        priority=1,
    )
    try:
        assert ebook_to_markdown(epub_bytes) == "RECOVERED"
        # ...unless the caller opted out of falling back
        with pytest.raises(EbookConversionError, match="kaboom"):
            ebook_to_markdown(epub_bytes, fallback=False)
    finally:
        del _ebook_backends["explodes"]
        del _ebook_backends["works"]


def test_all_backends_failing_reports_every_attempt(epub_bytes):
    """The aggregated error must name each backend and why it failed."""
    for i in (1, 2):
        register_ebook_backend(
            f"dud{i}",
            lambda src, fmt, i=i: (_ for _ in ()).throw(RuntimeError(f"dud{i} died")),
            is_available=lambda: True,
            formats=["zzz"],  # a format nothing else claims
            priority=i,
        )
    try:
        with pytest.raises(EbookConversionError) as excinfo:
            ebook_to_markdown(epub_bytes, input_format="zzz")
        message = str(excinfo.value)
        assert "dud1 died" in message and "dud2 died" in message
    finally:
        del _ebook_backends["dud1"]
        del _ebook_backends["dud2"]


def test_named_backend_failure_is_an_ebook_conversion_error(epub_bytes):
    """Callers catching EbookConversionError shouldn't see a backend's own type."""
    register_ebook_backend(
        "raises_valueerror",
        lambda src, fmt: (_ for _ in ()).throw(ValueError("raw")),
        is_available=lambda: True,
        formats=["epub"],
        priority=0,
    )
    try:
        with pytest.raises(EbookConversionError, match="raw"):
            ebook_to_markdown(epub_bytes, backend="raises_valueerror")
    finally:
        del _ebook_backends["raises_valueerror"]


def test_sniff_ebook_format(epub_bytes):
    assert sniff_ebook_format(epub_bytes) == "epub"
    assert sniff_ebook_format(b"\x00" * 60 + b"BOOKMOBI" + b"tail") == "mobi"
    assert sniff_ebook_format(b"just some text") is None


def test_register_ebook_converters_populates_a_registry():
    registry = {}
    register_ebook_converters(registry)
    assert AUTOWIRED_EBOOK_FORMATS <= set(registry)
    assert callable(registry["mobi"])


@pytest.mark.parametrize("ext", ["rb", "pdb", "opf", "prc", "snb", "tcr", "textile"])
def test_ambiguous_extensions_are_not_auto_claimed(ext):
    """.rb is Ruby far more often than Rocket eBook; don't hijack those files."""
    from dn.src import dflt_converters, _detect_content_type

    assert ext in EBOOK_FORMATS  # still convertible on explicit request
    assert ext not in AUTOWIRED_EBOOK_FORMATS
    assert ext not in dflt_converters
    assert _detect_content_type(b"class Foo\nend\n", f"x.{ext}") != ext


def test_source_files_do_not_get_routed_through_calibre(monkeypatch):
    """A Ruby file must convert as text, without spawning any external tool."""
    import dn.ebook
    from dn.src import bytes_to_markdown

    monkeypatch.setattr(
        dn.ebook,
        "_run",
        lambda *a, **kw: pytest.fail("an external converter was spawned for .rb"),
    )
    assert "class Foo" in bytes_to_markdown(b"class Foo\nend\n", key="script.rb")


def test_custom_format_backend_is_fully_wired():
    """Registering a backend for a NEW format must reach the whole pipeline."""
    register_ebook_backend(
        "cbz_reader",
        lambda src, fmt: "COMIC TEXT",
        is_available=lambda: True,
        formats=["cbz"],
        priority=0,
    )
    try:
        assert "cbz" in ebook_formats()
        registry = {}
        register_ebook_converters(registry)
        assert "cbz" in registry
        assert "cbz" in _no_backend_message("cbz")
    finally:
        del _ebook_backends["cbz_reader"]


def test_register_ebook_converters_does_not_clobber_without_force():
    registry = {"epub": "mine"}
    register_ebook_converters(registry)
    assert registry["epub"] == "mine"
    register_ebook_converters(registry, formats=["epub"], force=True)
    assert callable(registry["epub"])


def test_dn_dflt_converters_include_ebook_formats():
    """The whole point of the wiring: bytes_to_markdown must know about ebooks."""
    from dn.src import dflt_converters

    assert AUTOWIRED_EBOOK_FORMATS <= set(dflt_converters)


def test_unhandled_format_raises_a_helpful_error():
    with pytest.raises(EbookConversionError, match="No backend handles"):
        ebook_to_markdown(b"\x00" * 100, input_format="cbz")


def test_undeterminable_format_raises_a_helpful_error():
    with pytest.raises(EbookConversionError, match="Could not determine"):
        ebook_to_markdown(b"not an ebook at all")


def test_unknown_backend_name_raises():
    with pytest.raises(EbookConversionError, match="Unknown backend"):
        ebook_to_markdown(b"x", input_format="epub", backend="no-such-backend")


def test_check_ebook_requirements_reports_every_backend():
    report = check_ebook_requirements(verbose=False)
    assert set(report) == set(ebook_backends())
    assert report["calibre_pandoc"]["requires"] == ("calibre", "pandoc")
    for info in report.values():
        # Anything unavailable must tell the user how to fix that.
        assert info["available"] or info["install"]


# --------------------------------------------------------------------------------------
# Real conversions (skipped when the external tools aren't installed)


needs_a_backend = pytest.mark.skipif(
    not available_ebook_backends("epub"),
    reason="no ebook-to-markdown backend available (needs calibre, pandoc or ebooklib)",
)


@needs_a_backend
def test_epub_file_to_markdown(epub_file):
    """User story: I have an ebook on disk and want its text as markdown."""
    md = ebook_to_markdown(epub_file)
    assert _CHAPTER_TEXT in md
    assert _CHAPTER_TITLE in md


@needs_a_backend
def test_epub_bytes_to_markdown(epub_bytes):
    """User story: I have ebook bytes (from a store or a download), not a file."""
    md = ebook_to_markdown(epub_bytes, input_format="epub")
    assert _CHAPTER_TEXT in md


@needs_a_backend
def test_bytes_to_markdown_routes_ebooks(epub_bytes):
    """User story: I use dn's generic entry point and it just handles the ebook."""
    from dn import bytes_to_markdown

    assert _CHAPTER_TEXT in bytes_to_markdown(epub_bytes, key="tiny_book.epub")
    # ...and with no format hint at all, via magic-byte sniffing
    assert _CHAPTER_TEXT in bytes_to_markdown(epub_bytes)


@pytest.mark.parametrize(
    "backend", ["calibre_pandoc", "pandoc", "calibre_txt", "ebooklib"]
)
def test_each_available_backend_extracts_the_text(epub_bytes, backend):
    """Every backend, when installed, must produce the book's actual text."""
    if backend not in available_ebook_backends("epub"):
        pytest.skip(f"backend {backend!r} is not available on this machine")
    md = ebook_to_markdown(epub_bytes, input_format="epub", backend=backend)
    assert _CHAPTER_TEXT in md

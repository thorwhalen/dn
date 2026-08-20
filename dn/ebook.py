"""
Convert ebook formats (EPUB, MOBI, AZW3, FB2, LIT, ...) to markdown.

Ebooks are containers of styled HTML, so the quality of the markdown you get out
depends on two things: resolving the publisher's CSS (which is where emphasis and
structure usually live) and rendering the result without leaving HTML soup behind.
No single pure-python library does both well, so this module is organized as a
**backend registry**: several conversion strategies, each declaring what it needs
and which formats it handles, tried in preference order.

The simple case is a one-liner -- point it at a file and get markdown back:

>>> md = ebook_to_markdown('book.epub')  # doctest: +SKIP

The registry is open: inspect it, reorder it, or add your own strategy.

>>> 'calibre_pandoc' in ebook_backends()
True

Backends shipped, in default preference order:

``calibre_pandoc``
    Calibre's ``ebook-convert`` normalizes the ebook into DOCX -- which resolves
    CSS classes into real character formatting -- and then ``pandoc`` renders
    clean GitHub-flavored markdown. Best fidelity, and covers every format
    calibre reads.
``pandoc``
    Pandoc reading the ebook directly. No calibre needed, but limited to the
    formats pandoc understands, and it cannot resolve class-based emphasis, so
    italics and bold from CSS-styled ebooks are lost.
``calibre_txt``
    Calibre's own markdown-flavored TXT output. No pandoc needed, but it escapes
    punctuation aggressively and flattens nested blockquotes.
``ebooklib``
    Pure python (``ebooklib`` + ``html2text``). EPUB only, but needs no external
    binary.

Neither calibre nor pandoc is a python package. Call :func:`check_ebook_requirements`
for a report of what is available and how to install what is not.

>>> report = check_ebook_requirements(verbose=False)
>>> sorted(report) == sorted(ebook_backends())
True
"""

import importlib.util
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union
from collections.abc import Callable, Iterable, MutableMapping

from dn.util import is_url, url_to_contents

__all__ = [
    "EBOOK_FORMATS",
    "AUTOWIRED_EBOOK_FORMATS",
    "ebook_formats",
    "EbookBackend",
    "EbookConversionError",
    "ebook_to_markdown",
    "ebook_backends",
    "available_ebook_backends",
    "register_ebook_backend",
    "register_ebook_converters",
    "check_ebook_requirements",
    "sniff_ebook_format",
    "find_ebook_convert",
    "find_pandoc",
    "calibre_pandoc_to_markdown",
    "pandoc_to_markdown",
    "calibre_txt_to_markdown",
    "ebooklib_to_markdown",
]


# --------------------------------------------------------------------------------------
# Formats

#: Formats :func:`ebook_to_markdown` will attempt when asked explicitly.
#:
#: Deliberately excludes formats ``dn`` already converts natively (``pdf``,
#: ``docx``, ``html``, ``xlsx``, ``pptx``, ``ipynb``) so registering these
#: converters never shadows a lighter-weight one, and excludes image containers
#: (``cbz``, ``djvu``, ...) which hold no extractable text.
#:
#: Being in here does *not* mean the extension is auto-detected -- see
#: :data:`AUTOWIRED_EBOOK_FORMATS`.
EBOOK_FORMATS = frozenset(
    {
        "azw",
        "azw3",
        "azw4",
        "chm",
        "epub",
        "fb2",
        "fbz",
        "htmlz",
        "kepub",
        "lit",
        "lrf",
        "mobi",
        "odt",
        "opf",
        "pdb",
        "pml",
        "pmlz",
        "pobi",
        "prc",
        "rb",
        "rtf",
        "snb",
        "tcr",
        "textile",
        "txtz",
        "updb",
    }
)

#: The subset of :data:`EBOOK_FORMATS` wired into ``dn``'s converter registry
#: and filename-based content detection.
#:
#: Several extensions calibre reads are *predominantly* something else:
#: ``.rb`` is Ruby source far more often than Rocket eBook, ``.pdb`` is a
#: Protein Data Bank or debug-symbol file far more often than PalmDoc, ``.prc``
#: and ``.snb`` and ``.tcr`` are ambiguous, ``.opf`` is a manifest rather than a
#: book, and ``.textile`` is a markup language. Auto-claiming those would route
#: ordinary source files through calibre -- slow, and failing where the plain
#: text fallback used to succeed. They stay convertible on explicit request via
#: ``ebook_to_markdown(src, input_format='rb')``.
AUTOWIRED_EBOOK_FORMATS = frozenset(
    {
        "azw",
        "azw3",
        "azw4",
        "chm",
        "epub",
        "fb2",
        "fbz",
        "htmlz",
        "kepub",
        "lit",
        "lrf",
        "mobi",
        "odt",
        "pmlz",
        "pobi",
        "rtf",
        "txtz",
        "updb",
    }
)

#: Formats pandoc can read without calibre's help.
PANDOC_READABLE_FORMATS = frozenset({"epub", "fb2", "odt", "rtf", "textile", "docx"})

#: Pandoc output target. ``-raw_html`` drops the ``<span>``/``<div>`` scaffolding
#: that ebook toolchains leave behind, which is the difference between readable
#: markdown and HTML soup.
DFLT_PANDOC_TO = "gfm-raw_html"


class EbookConversionError(RuntimeError):
    """Raised when an ebook could not be converted to markdown."""


# --------------------------------------------------------------------------------------
# External tool discovery
#
# calibre and pandoc are ordinary executables, not python packages, so they can't
# be found through import machinery. We look on PATH first, then in the places
# each platform's installer actually puts them.

_CALIBRE_CANDIDATES = (
    "/Applications/calibre.app/Contents/MacOS/ebook-convert",
    "/Applications/Calibre.app/Contents/MacOS/ebook-convert",
    "/opt/calibre/ebook-convert",
    "/usr/bin/ebook-convert",
    r"C:\Program Files\Calibre2\ebook-convert.exe",
    r"C:\Program Files (x86)\Calibre2\ebook-convert.exe",
)

_PANDOC_CANDIDATES = (
    "/opt/homebrew/bin/pandoc",
    "/usr/local/bin/pandoc",
    "/usr/bin/pandoc",
    r"C:\Program Files\Pandoc\pandoc.exe",
)


def _importable(module_name: str) -> bool:
    """Whether ``module_name`` can be imported, without actually importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _find_executable(name: str, candidates: Iterable[str] = ()) -> str | None:
    """Find an executable on PATH, falling back to explicit candidate paths.

    Args:
        name: Executable name to look for on ``PATH`` (e.g. ``'pandoc'``).
        candidates: Absolute paths to try if ``PATH`` yields nothing.

    Returns:
        Path to the executable, or ``None`` if it could not be found.
    """
    path = shutil.which(name)
    if path:
        return path
    for candidate in candidates:
        candidate = os.path.expandvars(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_ebook_convert() -> str | None:
    """Find Calibre's ``ebook-convert`` binary, or ``None`` if it isn't installed.

    >>> find_ebook_convert()  # doctest: +SKIP
    '/Applications/calibre.app/Contents/MacOS/ebook-convert'
    """
    return _find_executable("ebook-convert", _CALIBRE_CANDIDATES)


def find_pandoc() -> str | None:
    """Find the ``pandoc`` binary, or ``None`` if it isn't installed.

    >>> find_pandoc()  # doctest: +SKIP
    '/opt/homebrew/bin/pandoc'
    """
    return _find_executable("pandoc", _PANDOC_CANDIDATES)


def _tail(text: str, *, n_lines: int = 6) -> str:
    """Last few meaningful lines of tool output.

    Calibre dumps a full python traceback on failure and puts the line that
    actually explains the problem at the very end, so the tail is the useful part.

    >>> _tail('a\\n\\nb\\nc\\n', n_lines=2)
    'b\\nc'
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-n_lines:])


def _run(
    cmd: list, *, what: str, timeout: Optional[float] = None
) -> subprocess.CompletedProcess:
    """Run a command, raising an :class:`EbookConversionError` that says what broke.

    ``stdin`` is closed: a tool that decides to prompt should fail rather than
    silently consume the caller's stdin and hang.
    """
    try:
        result = subprocess.run(
            cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise EbookConversionError(f"{what} timed out after {timeout}s") from e
    except OSError as e:
        raise EbookConversionError(f"Could not run {what}: {e}") from e
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"").decode("utf-8", "replace")
        raise EbookConversionError(
            f"{what} failed (exit code {result.returncode}): {_tail(detail)}"
        )
    return result


# --------------------------------------------------------------------------------------
# Conversion primitives
#
# Each is directly callable with its full set of options. The registry below only
# handles dispatch, so tuning a backend means calling these yourself (or
# registering a `partial` of one).


def _ebook_convert(
    src: Union[str, Path],
    dst: Union[str, Path],
    *,
    extra_args: Iterable[str] = (),
    ebook_convert_path: str | None = None,
) -> Path:
    """Run Calibre's ``ebook-convert`` on ``src``, writing ``dst``."""
    ebook_convert_path = ebook_convert_path or find_ebook_convert()
    if ebook_convert_path is None:
        raise EbookConversionError(
            "Calibre's ebook-convert is required but was not found. "
            + _install_hint("calibre")
        )
    _run(
        [ebook_convert_path, str(src), str(dst), *extra_args],
        what="Calibre's ebook-convert",
    )
    return Path(dst)


def _pandoc(
    src: Union[str, Path],
    *,
    from_format: str,
    to_format: str = DFLT_PANDOC_TO,
    extract_media: str | None = None,
    extra_args: Iterable[str] = (),
    pandoc_path: str | None = None,
) -> str:
    """Run pandoc on ``src`` and return its output as text."""
    pandoc_path = pandoc_path or find_pandoc()
    if pandoc_path is None:
        raise EbookConversionError(
            "pandoc is required but was not found. " + _install_hint("pandoc")
        )
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "out.md"
        cmd = [
            pandoc_path,
            "-f",
            from_format,
            "-t",
            to_format,
            "--wrap=none",
            str(src),
            "-o",
            str(out),
        ]
        if extract_media:
            cmd.append(f"--extract-media={extract_media}")
        cmd.extend(extra_args)
        _run(cmd, what="pandoc")
        return out.read_text(encoding="utf-8")


def calibre_pandoc_to_markdown(
    src: Union[str, Path],
    input_format: str | None = None,
    *,
    to_format: str = DFLT_PANDOC_TO,
    extract_media: str | None = None,
) -> str:
    """Convert an ebook to markdown via Calibre's DOCX output, rendered by pandoc.

    The DOCX detour is the point: calibre resolves the ebook's CSS into real
    character formatting, so italics and bold survive as ``*emphasis*`` instead of
    being dropped with the ``<span class="calibre3">`` wrappers that carried them.

    Args:
        src: Path to the ebook file.
        input_format: Ebook format. Unused here (calibre dispatches on the file
            extension); accepted so backends share one signature.
        to_format: Pandoc output target.
        extract_media: Directory to write embedded images to. If ``None``, image
            references are kept but the files are not extracted.

    Returns:
        Markdown text.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        docx = Path(tmp_dir) / "book.docx"
        _ebook_convert(src, docx)
        return _pandoc(
            docx,
            from_format="docx",
            to_format=to_format,
            extract_media=extract_media,
        )


def pandoc_to_markdown(
    src: Union[str, Path],
    input_format: str | None = None,
    *,
    to_format: str = DFLT_PANDOC_TO,
    extract_media: str | None = None,
) -> str:
    """Convert an ebook to markdown with pandoc alone (no calibre).

    Args:
        src: Path to the ebook file.
        input_format: Ebook format, used as pandoc's ``-f``. Inferred from the
            file extension when ``None``.
        to_format: Pandoc output target.
        extract_media: Directory to write embedded images to.

    Returns:
        Markdown text.
    """
    from_format = (input_format or Path(src).suffix.lstrip(".")).lower()
    return _pandoc(
        src,
        from_format=from_format,
        to_format=to_format,
        extract_media=extract_media,
    )


def calibre_txt_to_markdown(
    src: Union[str, Path], input_format: str | None = None
) -> str:
    """Convert an ebook to markdown using Calibre's markdown-flavored TXT output.

    A fallback for when pandoc isn't available. Calibre understands its own CSS so
    emphasis survives, but it escapes punctuation aggressively (``\\(`` for ``(``)
    and flattens nested blockquotes onto one line.

    Args:
        src: Path to the ebook file.
        input_format: Ebook format. Unused (calibre dispatches on the extension).

    Returns:
        Markdown text.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "book.txt"
        _ebook_convert(
            src,
            out,
            extra_args=(
                "--txt-output-formatting=markdown",
                "--keep-links",
                "--keep-image-references",
            ),
        )
        return out.read_text(encoding="utf-8", errors="replace")


def ebooklib_to_markdown(
    src: Union[str, Path],
    input_format: str | None = None,
    *,
    item_separator: str = "\n\n",
) -> str:
    """Convert an EPUB to markdown in pure python, via ``ebooklib`` and ``html2text``.

    Needs no external binary, but only reads EPUB, and cannot resolve class-based
    emphasis (the CSS is never applied), so styled italics and bold are lost.

    Args:
        src: Path to the EPUB file.
        input_format: Ebook format. Only ``'epub'`` is supported.
        item_separator: Text placed between successive document items.

    Returns:
        Markdown text.
    """
    try:
        import ebooklib  # pip install EbookLib
        from ebooklib import epub
        import html2text  # pip install html2text
    except ImportError as e:
        raise EbookConversionError(
            "The 'ebooklib' backend needs ebooklib and html2text. "
            + _install_hint("ebooklib")
        ) from e

    book = epub.read_epub(str(src))
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = False
    converter.body_width = 0  # don't hard-wrap

    chunks = (
        converter.handle(item.get_content().decode("utf-8", "replace"))
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    )
    return item_separator.join(chunk.strip() for chunk in chunks if chunk.strip())


# --------------------------------------------------------------------------------------
# Backend registry


@dataclass(frozen=True)
class EbookBackend:
    """A strategy for turning an ebook file into markdown.

    Attributes:
        name: Registry key.
        convert: ``(src_path, input_format) -> markdown``.
        is_available: ``() -> bool``, whether this backend's requirements are met.
        formats: Formats handled, or ``None`` for "everything in EBOOK_FORMATS".
        priority: Backends are tried in ascending priority (lowest first).
        requires: Human-readable requirement names, used by
            :func:`check_ebook_requirements`.
    """

    name: str
    convert: Callable[..., str]
    is_available: Callable[[], bool]
    formats: Optional[frozenset] = None
    priority: int = 50
    requires: tuple = ()

    def handles(self, input_format: str) -> bool:
        """Whether this backend claims support for ``input_format``.

        >>> backend = EbookBackend('x', lambda s, f: '', lambda: True)
        >>> backend.handles('mobi'), backend.handles('cbz')
        (True, False)
        """
        formats = EBOOK_FORMATS if self.formats is None else self.formats
        return input_format.lower() in formats


_ebook_backends: dict = {}


def register_ebook_backend(
    name: str,
    convert: Callable[..., str],
    *,
    is_available: Callable[[], bool],
    formats: Optional[Iterable[str]] = None,
    priority: int = 50,
    requires: Iterable[str] = (),
    force: bool = False,
) -> EbookBackend:
    """Register an ebook-to-markdown backend.

    Args:
        name: Registry key. Re-registering an existing name needs ``force=True``.
        convert: ``(src_path, input_format) -> markdown``.
        is_available: ``() -> bool``, checked before the backend is tried.
        formats: Formats handled. ``None`` means all of :data:`EBOOK_FORMATS`.
        priority: Backends are tried in ascending priority (lowest first).
        requires: Human-readable requirement names, for the requirements report.
        force: Overwrite an existing registration under the same name.

    Returns:
        The registered :class:`EbookBackend`.

    Raises:
        ValueError: If ``name`` is already registered and ``force`` is False.

    Example:

    >>> backend = register_ebook_backend(
    ...     'shouty',
    ...     lambda src, fmt: 'HELLO',
    ...     is_available=lambda: True,
    ...     formats=['epub'],
    ...     priority=99,
    ... )
    >>> try:
    ...     backend.handles('epub'), backend.handles('mobi')
    ... finally:
    ...     _ = _ebook_backends.pop('shouty')
    (True, False)
    """
    if name in _ebook_backends and not force:
        raise ValueError(
            f"Backend {name!r} is already registered. Pass force=True to replace it."
        )
    backend = EbookBackend(
        name=name,
        convert=convert,
        is_available=is_available,
        formats=None if formats is None else frozenset(f.lower() for f in formats),
        priority=priority,
        requires=tuple(requires),
    )
    _ebook_backends[name] = backend
    return backend


def ebook_backends(input_format: str | None = None) -> tuple:
    """Names of all registered backends, in the order they'd be tried.

    Args:
        input_format: If given, restrict to backends that handle that format.

    Returns:
        Backend names, best first.

    >>> ebook_backends()[0]
    'calibre_pandoc'
    >>> 'ebooklib' in ebook_backends('mobi')  # ebooklib is EPUB-only
    False
    """
    return tuple(b.name for b in _sorted_backends(input_format))


def available_ebook_backends(input_format: str | None = None) -> tuple:
    """Names of registered backends whose requirements are actually met.

    Args:
        input_format: If given, restrict to backends that handle that format.

    Returns:
        Backend names, best first. Empty if nothing is installed.

    >>> available_ebook_backends()  # doctest: +SKIP
    ('calibre_pandoc', 'pandoc', 'calibre_txt')
    """
    return tuple(
        b.name for b in _sorted_backends(input_format) if _safe_is_available(b)
    )


def ebook_formats() -> frozenset:
    """Every format some registered backend claims, right now.

    :data:`EBOOK_FORMATS` is the built-in baseline; registering a backend with
    an explicit ``formats`` adds to it. Derived live rather than snapshotted, so
    a custom backend for a new format is visible to
    :func:`register_ebook_converters` and to error messages.

    >>> 'epub' in ebook_formats()
    True
    """
    extra = (b.formats for b in _ebook_backends.values() if b.formats is not None)
    return EBOOK_FORMATS.union(*extra) if _ebook_backends else EBOOK_FORMATS


def _sorted_backends(input_format: str | None = None) -> list:
    """Registered backends in try-order, optionally filtered by format."""
    backends = _ebook_backends.values()
    if input_format is not None:
        backends = [b for b in backends if b.handles(input_format)]
    return sorted(backends, key=lambda b: (b.priority, b.name))


def _safe_is_available(backend: EbookBackend) -> bool:
    """``backend.is_available()``, treating a raising check as "not available"."""
    try:
        return bool(backend.is_available())
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Format sniffing


def _zip_declares_epub(data: bytes) -> bool:
    """Whether ``data`` is a zip whose ``mimetype`` entry declares an EPUB."""
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return zf.read("mimetype").strip() == b"application/epub+zip"
    except (zipfile.BadZipFile, KeyError, OSError, RuntimeError):
        return False


def sniff_ebook_format(data: bytes) -> str | None:
    """Guess an ebook format from its leading bytes.

    Recognizes EPUB (a zip whose first entry declares the epub mimetype) and the
    MOBI/AZW family (a PalmDB header whose type field is ``BOOKMOBI``).

    Args:
        data: Leading bytes of the file (at least 68 bytes to detect MOBI).

    Returns:
        Format name (e.g. ``'epub'``), or ``None`` if unrecognized.

    Examples:

    >>> sniff_ebook_format(b'not an ebook') is None
    True
    >>> sniff_ebook_format(b'\\x00' * 60 + b'BOOKMOBI' + b'rest')
    'mobi'
    """
    if data[:4] == b"PK\x03\x04":
        # The spec puts an uncompressed "mimetype" entry first, so the cheap
        # substring check catches conforming files. Plenty of real EPUBs violate
        # that, so fall back to actually reading the entry.
        if b"application/epub+zip" in data[:100] or _zip_declares_epub(data):
            return "epub"
    if data[60:68] in (b"BOOKMOBI", b"TEXtREAd"):
        # The MOBI/AZW/AZW3 family shares this PalmDB type. Calibre reads them all
        # through the same code path, so reporting 'mobi' is enough to convert.
        return "mobi"
    if data[:5] == b"<?xml" and b"<FictionBook" in data[:2048]:
        return "fb2"
    return None


# --------------------------------------------------------------------------------------
# The main entry point


def ebook_to_markdown(
    src: Union[bytes, str, Path],
    *,
    input_format: str | None = None,
    backend: str | None = None,
    fallback: bool = True,
) -> str:
    """Convert an ebook to markdown.

    Accepts whatever you have on hand -- raw bytes, a file path, or a URL -- and
    picks the best available backend unless you name one.

    Args:
        src: Ebook bytes, a path to an ebook file, or a URL to fetch one from.
        input_format: Ebook format (e.g. ``'mobi'``). Inferred from the file
            extension, then from the leading bytes, when not given.
        backend: Name of the backend to use. When ``None``, backends are tried in
            preference order (see :func:`ebook_backends`).
        fallback: If a chosen backend fails mid-conversion, try the next available
            one. Set ``False`` to let the first failure propagate.

    Returns:
        Markdown text.

    Raises:
        EbookConversionError: If the format can't be determined, no backend is
            available, or every attempted backend failed. The message names the
            missing requirements.

    Examples:

    >>> ebook_to_markdown('book.epub')  # doctest: +SKIP
    '# Chapter One\\n\\n...'
    >>> ebook_to_markdown(mobi_bytes, input_format='mobi')  # doctest: +SKIP
    '...'
    """
    src_bytes, resolved_format = _resolve_src(src, input_format)

    if resolved_format is None:
        raise EbookConversionError(
            "Could not determine the ebook format. Pass input_format explicitly, "
            f"e.g. input_format='epub'. Known formats: {sorted(EBOOK_FORMATS)}"
        )
    resolved_format = resolved_format.lower()

    if backend is not None:
        if backend not in _ebook_backends:
            raise EbookConversionError(
                f"Unknown backend {backend!r}. Registered: {ebook_backends()}"
            )
        candidates = [_ebook_backends[backend]]
    else:
        candidates = [
            b for b in _sorted_backends(resolved_format) if _safe_is_available(b)
        ]

    if not candidates:
        raise EbookConversionError(_no_backend_message(resolved_format))

    with tempfile.TemporaryDirectory() as tmp_dir:
        src_path = Path(tmp_dir) / f"book.{resolved_format}"
        src_path.write_bytes(src_bytes)

        errors = []
        for candidate in candidates:
            try:
                return candidate.convert(src_path, resolved_format)
            except Exception as e:
                errors.append(f"  {candidate.name}: {e}")
                if not fallback or backend is not None:
                    raise EbookConversionError(
                        f"Backend {candidate.name!r} failed converting "
                        f"{resolved_format!r} to markdown: {e}"
                    ) from e

    tried = "\n".join(errors)
    raise EbookConversionError(
        f"All backends failed converting {resolved_format!r} to markdown:\n{tried}"
    )


def _resolve_src(src, input_format: str | None):
    """Normalize ``src`` to ``(bytes, format)``, inferring the format if it can."""
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
        return data, input_format or sniff_ebook_format(data)

    src = os.path.expanduser(str(src))
    if is_url(src):
        data = url_to_contents(src)
        fmt = input_format or Path(src.split("?", 1)[0]).suffix.lstrip(".").lower()
        return data, fmt or sniff_ebook_format(data)

    data = Path(src).read_bytes()
    fmt = input_format or Path(src).suffix.lstrip(".").lower()
    return data, fmt or sniff_ebook_format(data)


def _no_backend_message(input_format: str) -> str:
    """Explain that nothing can convert ``input_format``, and how to fix that."""
    handlers = _sorted_backends(input_format)
    if not handlers:
        return (
            f"No backend handles the {input_format!r} format. "
            f"Formats with backends: {sorted(ebook_formats())}"
        )
    needed = sorted({req for b in handlers for req in b.requires})
    hints = "\n".join(f"  - {_install_hint(req)}" for req in needed)
    return (
        f"No available backend for {input_format!r}: every backend that handles it "
        f"({', '.join(b.name for b in handlers)}) is missing its requirements.\n"
        f"Install one of:\n{hints}"
    )


# --------------------------------------------------------------------------------------
# Requirements reporting


_INSTALL_HINTS = {
    "calibre": (
        "Calibre (provides ebook-convert) -- https://calibre-ebook.com/download\n"
        "      macOS: brew install --cask calibre  |  "
        "Linux: sudo apt install calibre  |  Windows: winget install calibre.calibre"
    ),
    "pandoc": (
        "pandoc -- https://pandoc.org/installing.html\n"
        "      macOS: brew install pandoc  |  "
        "Linux: sudo apt install pandoc  |  Windows: winget install JohnMacFarlane.Pandoc"
    ),
    "ebooklib": "pip install 'dn[ebook]'  (installs EbookLib and html2text)",
}


def _install_hint(requirement: str) -> str:
    """Installation guidance for a named requirement."""
    return _INSTALL_HINTS.get(requirement, f"install {requirement}")


def check_ebook_requirements(*, verbose: bool = True) -> dict:
    """Report which ebook backends are usable, and how to install the missing ones.

    Args:
        verbose: Print a human-readable report as well as returning it.

    Returns:
        ``{backend_name: {'available', 'requires', 'formats', 'install'}}``.

    Example:

    >>> report = check_ebook_requirements(verbose=False)
    >>> report['calibre_pandoc']['requires']
    ('calibre', 'pandoc')
    """
    report = {}
    for backend in _sorted_backends():
        available = _safe_is_available(backend)
        report[backend.name] = {
            "available": available,
            "requires": backend.requires,
            "formats": (
                "all" if backend.formats is None else tuple(sorted(backend.formats))
            ),
            "install": (
                None if available else [_install_hint(req) for req in backend.requires]
            ),
        }

    if verbose:
        print("Ebook-to-markdown backends (best first):\n")
        for name, info in report.items():
            mark = "OK     " if info["available"] else "MISSING"
            print(
                f"  [{mark}] {name}  (needs: {', '.join(info['requires']) or 'nothing'})"
            )
            for hint in info["install"] or ():
                print(f"      {hint}")
        if not any(info["available"] for info in report.values()):
            print("\nNo backend is available: ebook conversion will not work yet.")

    return report


# --------------------------------------------------------------------------------------
# Register the shipped backends
#
# Priorities encode the fidelity ranking documented in the module docstring.

register_ebook_backend(
    "calibre_pandoc",
    calibre_pandoc_to_markdown,
    is_available=lambda: find_ebook_convert() is not None and find_pandoc() is not None,
    priority=10,
    requires=("calibre", "pandoc"),
)

register_ebook_backend(
    "pandoc",
    pandoc_to_markdown,
    is_available=lambda: find_pandoc() is not None,
    formats=PANDOC_READABLE_FORMATS,
    priority=20,
    requires=("pandoc",),
)

register_ebook_backend(
    "calibre_txt",
    calibre_txt_to_markdown,
    is_available=lambda: find_ebook_convert() is not None,
    priority=30,
    requires=("calibre",),
)

register_ebook_backend(
    "ebooklib",
    ebooklib_to_markdown,
    is_available=lambda: _importable("ebooklib") and _importable("html2text"),
    formats=("epub",),
    priority=40,
    requires=("ebooklib",),
)


# --------------------------------------------------------------------------------------
# Wiring into dn's converter registry


def register_ebook_converters(
    converters: MutableMapping,
    *,
    formats: Optional[Iterable[str]] = None,
    force: bool = False,
) -> MutableMapping:
    """Add ``format -> (bytes -> markdown)`` entries for ebook formats.

    Used to wire this module into ``dn.src.dflt_converters`` so that
    ``bytes_to_markdown`` handles ebooks. Entries are registered whether or not a
    backend is installed: calling one without calibre or pandoc raises an
    :class:`EbookConversionError` naming what to install, which beats silently
    falling through to a base64 dump of the file.

    Args:
        converters: The registry to add to.
        formats: Formats to register.
        force: Replace existing entries for these formats.

    Returns:
        The (mutated) ``converters`` mapping.

    Example:

    >>> registry = {}
    >>> _ = register_ebook_converters(registry)
    >>> 'mobi' in registry and 'epub' in registry
    True
    """
    if formats is None:
        # The autowired baseline, plus any format a custom backend has claimed
        # since import (those are opt-in by definition, so no ambiguity risk).
        formats = AUTOWIRED_EBOOK_FORMATS | (ebook_formats() - EBOOK_FORMATS)
    for fmt in formats:
        fmt = fmt.lower()
        if force or fmt not in converters:
            converters[fmt] = partial(ebook_to_markdown, input_format=fmt)
    return converters

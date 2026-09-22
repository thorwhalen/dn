# dn.ebook

Convert ebook formats (EPUB, MOBI, AZW3, FB2, LIT, …) to markdown.

Ebooks are containers of styled HTML, so the quality of the markdown you get out
depends on two things: resolving the publisher’s CSS (which is where emphasis and
structure usually live) and rendering the result without leaving HTML soup behind.
No single pure-python library does both well, so this module is organized as a
**backend registry**: several conversion strategies, each declaring what it needs
and which formats it handles, tried in preference order.

The simple case is a one-liner – point it at a file and get markdown back:

```pycon
>>> md = ebook_to_markdown('book.epub')
```

The registry is open: inspect it, reorder it, or add your own strategy.

```pycon
>>> 'calibre_pandoc' in ebook_backends()
True
```

Backends shipped, in default preference order:

`calibre_pandoc`
: Calibre’s `ebook-convert` normalizes the ebook into DOCX – which resolves
  CSS classes into real character formatting – and then `pandoc` renders
  clean GitHub-flavored markdown. Best fidelity, and covers every format
  calibre reads.

`pandoc`
: Pandoc reading the ebook directly. No calibre needed, but limited to the
  formats pandoc understands, and it cannot resolve class-based emphasis, so
  italics and bold from CSS-styled ebooks are lost.

`calibre_txt`
: Calibre’s own markdown-flavored TXT output. No pandoc needed, but it escapes
  punctuation aggressively and flattens nested blockquotes.

`ebooklib`
: Pure python (`ebooklib` + `markdownify`). EPUB only, but needs no external
  binary.

Neither calibre nor pandoc is a python package. Call [`check_ebook_requirements()`](#dn.ebook.check_ebook_requirements)
for a report of what is available and how to install what is not.

```pycon
>>> report = check_ebook_requirements(verbose=False)
>>> sorted(report) == sorted(ebook_backends())
True
```

### Module Attributes

| [`EBOOK_FORMATS`](#dn.ebook.EBOOK_FORMATS)           | Formats [`ebook_to_markdown()`](#dn.ebook.ebook_to_markdown) will attempt when asked explicitly.                                        |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`AUTOWIRED_EBOOK_FORMATS`](#dn.ebook.AUTOWIRED_EBOOK_FORMATS) | The subset of [`EBOOK_FORMATS`](#dn.ebook.EBOOK_FORMATS) wired into `dn`'s converter registry and filename-based content detection. |

### Functions

| [`ebook_formats`](#dn.ebook.ebook_formats)()                                  | Every format some registered backend claims, right now.                       |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`ebook_to_markdown`](#dn.ebook.ebook_to_markdown)(src, \*[, input_format, ...])  | Convert an ebook to markdown.                                                 |
| [`ebook_backends`](#dn.ebook.ebook_backends)([input_format])                   | Names of all registered backends, in the order they'd be tried.               |
| [`available_ebook_backends`](#dn.ebook.available_ebook_backends)([input_format])         | Names of registered backends whose requirements are actually met.             |
| [`register_ebook_backend`](#dn.ebook.register_ebook_backend)(name, convert, \*, ...)   | Register an ebook-to-markdown backend.                                        |
| [`register_ebook_converters`](#dn.ebook.register_ebook_converters)(converters, \*[, ...]) | Add `format -> (bytes -> markdown)` entries for ebook formats.                |
| [`check_ebook_requirements`](#dn.ebook.check_ebook_requirements)(\*[, verbose])          | Report which ebook backends are usable, and how to install the missing ones.  |
| [`sniff_ebook_format`](#dn.ebook.sniff_ebook_format)(data)                         | Guess an ebook format from its leading bytes.                                 |
| [`find_ebook_convert`](#dn.ebook.find_ebook_convert)()                             | Find Calibre's `ebook-convert` binary, or `None` if it isn't installed.       |
| [`find_pandoc`](#dn.ebook.find_pandoc)()                                    | Find the `pandoc` binary, or `None` if it isn't installed.                    |
| [`calibre_pandoc_to_markdown`](#dn.ebook.calibre_pandoc_to_markdown)(src[, ...])           | Convert an ebook to markdown via Calibre's DOCX output, rendered by pandoc.   |
| [`pandoc_to_markdown`](#dn.ebook.pandoc_to_markdown)(src[, input_format, ...])     | Convert an ebook to markdown with pandoc alone (no calibre).                  |
| [`calibre_txt_to_markdown`](#dn.ebook.calibre_txt_to_markdown)(src[, input_format])     | Convert an ebook to markdown using Calibre's markdown-flavored TXT output.    |
| [`ebooklib_to_markdown`](#dn.ebook.ebooklib_to_markdown)(src[, input_format, ...])   | Convert an EPUB to markdown in pure python, via `ebooklib` and `markdownify`. |

### Classes

| [`EbookBackend`](#dn.ebook.EbookBackend)(name, convert, is_available[, ...])   | A strategy for turning an ebook file into markdown.   |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------|

### Exceptions

| [`EbookConversionError`](#dn.ebook.EbookConversionError)   | Raised when an ebook could not be converted to markdown.   |
|-------------------------------------------------------------------------|------------------------------------------------------------|

### dn.ebook.AUTOWIRED_EBOOK_FORMATS *= frozenset({'azw', 'azw3', 'azw4', 'chm', 'epub', 'fb2', 'fbz', 'htmlz', 'kepub', 'lit', 'lrf', 'mobi', 'odt', 'pmlz', 'pobi', 'rtf', 'txtz', 'updb'})*

The subset of [`EBOOK_FORMATS`](#dn.ebook.EBOOK_FORMATS) wired into `dn`’s converter registry
and filename-based content detection.

Several extensions calibre reads are *predominantly* something else:
`.rb` is Ruby source far more often than Rocket eBook, `.pdb` is a
Protein Data Bank or debug-symbol file far more often than PalmDoc, `.prc`
and `.snb` and `.tcr` are ambiguous, `.opf` is a manifest rather than a
book, and `.textile` is a markup language. Auto-claiming those would route
ordinary source files through calibre – slow, and failing where the plain
text fallback used to succeed. They stay convertible on explicit request via
`ebook_to_markdown(src, input_format='rb')`.

### dn.ebook.EBOOK_FORMATS *= frozenset({'azw', 'azw3', 'azw4', 'chm', 'epub', 'fb2', 'fbz', 'htmlz', 'kepub', 'lit', 'lrf', 'mobi', 'odt', 'opf', 'pdb', 'pml', 'pmlz', 'pobi', 'prc', 'rb', 'rtf', 'snb', 'tcr', 'textile', 'txtz', 'updb'})*

Formats [`ebook_to_markdown()`](#dn.ebook.ebook_to_markdown) will attempt when asked explicitly.

Deliberately excludes formats `dn` already converts natively (`pdf`,
`docx`, `html`, `xlsx`, `pptx`, `ipynb`) so registering these
converters never shadows a lighter-weight one, and excludes image containers
(`cbz`, `djvu`, …) which hold no extractable text.

Being in here does *not* mean the extension is auto-detected – see
[`AUTOWIRED_EBOOK_FORMATS`](#dn.ebook.AUTOWIRED_EBOOK_FORMATS).

### *class* dn.ebook.EbookBackend(name, convert, is_available, formats=None, priority=50, requires=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A strategy for turning an ebook file into markdown.

#### name

Registry key.

#### convert

`(src_path, input_format) -> markdown`.

#### is_available

`() -> bool`, whether this backend’s requirements are met.

#### formats

Formats handled, or `None` for “everything in EBOOK_FORMATS”.

#### priority

Backends are tried in ascending priority (lowest first).

#### requires

Human-readable requirement names, used by
[`check_ebook_requirements()`](#dn.ebook.check_ebook_requirements).

#### handles(input_format)

Whether this backend claims support for `input_format`.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> backend = EbookBackend('x', lambda s, f: '', lambda: True)
>>> backend.handles('mobi'), backend.handles('cbz')
(True, False)
```

### *exception* dn.ebook.EbookConversionError

Bases: [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

Raised when an ebook could not be converted to markdown.

### dn.ebook.available_ebook_backends(input_format=None)

Names of registered backends whose requirements are actually met.

* **Parameters:**
  **input_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If given, restrict to backends that handle that format.
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)
* **Returns:**
  Backend names, best first. Empty if nothing is installed.

```pycon
>>> available_ebook_backends()
('calibre_pandoc', 'pandoc', 'calibre_txt')
```

### dn.ebook.calibre_pandoc_to_markdown(src, input_format=None, , to_format='gfm-raw_html', extract_media=None)

Convert an ebook to markdown via Calibre’s DOCX output, rendered by pandoc.

The DOCX detour is the point: calibre resolves the ebook’s CSS into real
character formatting, so italics and bold survive as `*emphasis*` instead of
being dropped with the `<span class="calibre3">` wrappers that carried them.

* **Parameters:**
  * **src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the ebook file.
  * **input_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ebook format. Unused here (calibre dispatches on the file
    extension); accepted so backends share one signature.
  * **to_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Pandoc output target.
  * **extract_media** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Directory to write embedded images to. If `None`, image
    references are kept but the files are not extracted.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Markdown text.

### dn.ebook.calibre_txt_to_markdown(src, input_format=None)

Convert an ebook to markdown using Calibre’s markdown-flavored TXT output.

A fallback for when pandoc isn’t available. Calibre understands its own CSS so
emphasis survives, but it escapes punctuation aggressively (`\(` for `(`)
and flattens nested blockquotes onto one line.

* **Parameters:**
  * **src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the ebook file.
  * **input_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ebook format. Unused (calibre dispatches on the extension).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Markdown text.

### dn.ebook.check_ebook_requirements(, verbose=True)

Report which ebook backends are usable, and how to install the missing ones.

* **Parameters:**
  **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print a human-readable report as well as returning it.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  `{backend_name: {'available', 'requires', 'formats', 'install'}}`.

### Example

```pycon
>>> report = check_ebook_requirements(verbose=False)
>>> report['calibre_pandoc']['requires']
('calibre', 'pandoc')
```

### dn.ebook.ebook_backends(input_format=None)

Names of all registered backends, in the order they’d be tried.

* **Parameters:**
  **input_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If given, restrict to backends that handle that format.
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)
* **Returns:**
  Backend names, best first.

```pycon
>>> ebook_backends()[0]
'calibre_pandoc'
>>> 'ebooklib' in ebook_backends('mobi')  # ebooklib is EPUB-only
False
```

### dn.ebook.ebook_formats()

Every format some registered backend claims, right now.

[`EBOOK_FORMATS`](#dn.ebook.EBOOK_FORMATS) is the built-in baseline; registering a backend with
an explicit `formats` adds to it. Derived live rather than snapshotted, so
a custom backend for a new format is visible to
[`register_ebook_converters()`](#dn.ebook.register_ebook_converters) and to error messages.

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)

```pycon
>>> 'epub' in ebook_formats()
True
```

### dn.ebook.ebook_to_markdown(src, , input_format=None, backend=None, fallback=True)

Convert an ebook to markdown.

Accepts whatever you have on hand – raw bytes, a file path, or a URL – and
picks the best available backend unless you name one.

* **Parameters:**
  * **src** (`Union`[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Ebook bytes, a path to an ebook file, or a URL to fetch one from.
  * **input_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ebook format (e.g. `'mobi'`). Inferred from the file
    extension, then from the leading bytes, when not given.
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Name of the backend to use. When `None`, backends are tried in
    preference order (see [`ebook_backends()`](#dn.ebook.ebook_backends)).
  * **fallback** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If a chosen backend fails mid-conversion, try the next available
    one. Set `False` to let the first failure propagate.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Markdown text.
* **Raises:**
  [**EbookConversionError**](#dn.ebook.EbookConversionError) – If the format can’t be determined, no backend is
      available, or every attempted backend failed. The message names the
      missing requirements.

### Examples

```pycon
>>> ebook_to_markdown('book.epub')
'# Chapter One\n\n...'
>>> ebook_to_markdown(mobi_bytes, input_format='mobi')
'...'
```

### dn.ebook.ebooklib_to_markdown(src, input_format=None, , item_separator='\\\\n\\\\n')

Convert an EPUB to markdown in pure python, via `ebooklib` and `markdownify`.

Needs no external binary, but only reads EPUB, and cannot resolve class-based
emphasis (the CSS is never applied), so styled italics and bold are lost.

* **Parameters:**
  * **src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the EPUB file.
  * **input_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ebook format. Only `'epub'` is supported.
  * **item_separator** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text placed between successive document items.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Markdown text.

### dn.ebook.find_ebook_convert()

Find Calibre’s `ebook-convert` binary, or `None` if it isn’t installed.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> find_ebook_convert()
'/Applications/calibre.app/Contents/MacOS/ebook-convert'
```

### dn.ebook.find_pandoc()

Find the `pandoc` binary, or `None` if it isn’t installed.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> find_pandoc()
'/opt/homebrew/bin/pandoc'
```

### dn.ebook.pandoc_to_markdown(src, input_format=None, , to_format='gfm-raw_html', extract_media=None)

Convert an ebook to markdown with pandoc alone (no calibre).

* **Parameters:**
  * **src** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the ebook file.
  * **input_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ebook format, used as pandoc’s `-f`. Inferred from the
    file extension when `None`.
  * **to_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Pandoc output target.
  * **extract_media** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Directory to write embedded images to.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Markdown text.

### dn.ebook.register_ebook_backend(name, convert, , is_available, formats=None, priority=50, requires=(), force=False)

Register an ebook-to-markdown backend.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Registry key. Re-registering an existing name needs `force=True`.
  * **convert** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – `(src_path, input_format) -> markdown`.
  * **is_available** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – `() -> bool`, checked before the backend is tried.
  * **formats** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Formats handled. `None` means all of [`EBOOK_FORMATS`](#dn.ebook.EBOOK_FORMATS).
  * **priority** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Backends are tried in ascending priority (lowest first).
  * **requires** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Human-readable requirement names, for the requirements report.
  * **force** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Overwrite an existing registration under the same name.
* **Return type:**
  [`EbookBackend`](#dn.ebook.EbookBackend)
* **Returns:**
  The registered [`EbookBackend`](#dn.ebook.EbookBackend).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `name` is already registered and `force` is False.

### Example

```pycon
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
```

### dn.ebook.register_ebook_converters(converters, , formats=None, force=False)

Add `format -> (bytes -> markdown)` entries for ebook formats.

Used to wire this module into `dn.src.dflt_converters` so that
`bytes_to_markdown` handles ebooks. Entries are registered whether or not a
backend is installed: calling one without calibre or pandoc raises an
[`EbookConversionError`](#dn.ebook.EbookConversionError) naming what to install, which beats silently
falling through to a base64 dump of the file.

* **Parameters:**
  * **converters** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)) – The registry to add to.
  * **formats** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Formats to register.
  * **force** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Replace existing entries for these formats.
* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)
* **Returns:**
  The (mutated) `converters` mapping.

### Example

```pycon
>>> registry = {}
>>> _ = register_ebook_converters(registry)
>>> 'mobi' in registry and 'epub' in registry
True
```

### dn.ebook.sniff_ebook_format(data)

Guess an ebook format from its leading bytes.

Recognizes EPUB (a zip whose first entry declares the epub mimetype) and the
MOBI/AZW family (a PalmDB header whose type field is `BOOKMOBI`).

* **Parameters:**
  **data** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – Leading bytes of the file (at least 68 bytes to detect MOBI).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  Format name (e.g. `'epub'`), or `None` if unrecognized.

### Examples

```pycon
>>> sniff_ebook_format(b'not an ebook') is None
True
>>> sniff_ebook_format(b'\x00' * 60 + b'BOOKMOBI' + b'rest')
'mobi'
```

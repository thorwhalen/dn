# dn.repair

Markdown repair and cleanup utilities.

Pure string→string transforms for fixing common markdown artifacts.
These arise from web scraping, AI-generated text, format conversion,
or copy-paste from rich-text sources.

Main entry point:

```default
>>> from dn.repair import repair_markdown
>>> repair_markdown('[Title\n\nDescription.](http://x.com)')
'[Title](http://x.com) — Description.'
```

Individual fixers can be used standalone or composed.

### Functions

| [`fix_empty_links`](#dn.repair.fix_empty_links)(md)                              | Remove empty links like `[](url)`.                                          |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`fix_multiline_links`](#dn.repair.fix_multiline_links)(md)                          | Fix markdown links whose text spans multiple lines.                         |
| [`remove_hyperlink_crap`](#dn.repair.remove_hyperlink_crap)(string)                    | Remove unwanted hyperlink artifacts from text.                              |
| [`remove_improperly_double_newlines`](#dn.repair.remove_improperly_double_newlines)(string)        | Remove spurious double newlines caused by whitespace-only lines.            |
| [`repair_markdown`](#dn.repair.repair_markdown)(md)                              | Fix common markdown rendering issues.                                       |
| [`strip_repeated_lines`](#dn.repair.strip_repeated_lines)(md, \*[, min_repeats, ...]) | Remove boilerplate lines that recur throughout scraped/aggregated markdown. |

### dn.repair.fix_empty_links(md)

Remove empty links like `[](url)`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> fix_empty_links('before [](http://x.com) after')
'before  after'
>>> fix_empty_links('[real](http://x.com)')
'[real](http://x.com)'
```

### dn.repair.fix_multiline_links(md)

Fix markdown links whose text spans multiple lines.

Collapses `[title\ndescription](url)` into
`[title](url) — description`.

* **Parameters:**
  **md** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Markdown text.
* **Returns:**
  Text with multi-line links collapsed.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> fix_multiline_links('[Hello\n\nWorld](http://x.com)')
'[Hello](http://x.com) — World'
>>> fix_multiline_links('[Single line](http://x.com)')
'[Single line](http://x.com)'
```

### dn.repair.remove_hyperlink_crap(string)

Remove unwanted hyperlink artifacts from text.

Cleans up common artifacts from AI-generated or copy-pasted text:

- ChatGPT UTM tracking: `?utm_source=chatgpt.com`
- ChatGPT citation markers: `oai_citation:\d+‡`
- Double hyperlinks: `[[X](Y)](Y)` → `[X](Y)`
  (common when copying from Claude)

* **Parameters:**
  **string** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to clean.
* **Returns:**
  Cleaned text.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> remove_hyperlink_crap('See [here](http://x.com?utm_source=chatgpt.com)')
'See [here](http://x.com)'
>>> remove_hyperlink_crap('[[Title](http://x.com)](http://x.com)')
'[Title](http://x.com)'
```

### dn.repair.remove_improperly_double_newlines(string)

Remove spurious double newlines caused by whitespace-only lines.

Replaces patterns like `\n   \n` (newline, spaces, newline) with a
single newline. Also normalizes `\r\n` and `\n\r` to `\n`.

* **Parameters:**
  **string** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to clean.
* **Returns:**
  Text with spurious double newlines removed.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> remove_improperly_double_newlines("a\n\nb\n  \nc")
'a\n\nb\nc'
```

### dn.repair.repair_markdown(md)

Fix common markdown rendering issues.

Applies all available fixers in sequence:

1. **Multi-line links**: `[text with\nnewlines](url)` → single-line link
   with description appended after an em-dash.
2. **Empty links**: `[](url)` → removed.

This is useful as a post-processing step on any markdown, regardless of
how it was generated.

* **Parameters:**
  **md** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Markdown text to repair.
* **Returns:**
  Repaired markdown text.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> repair_markdown('[Title\n\nDescription.](http://x.com)')
'[Title](http://x.com) — Description.'
>>> repair_markdown('[Good link](http://x.com)')
'[Good link](http://x.com)'
>>> repair_markdown('before [](http://empty) after')
'before  after'
```

### dn.repair.strip_repeated_lines(md, \*, min_repeats=30, protect=<function \_dflt_protect_repeated>, keep_first=False)

Remove boilerplate lines that recur throughout scraped/aggregated markdown.

Web-scrape and multi-page aggregates are dominated by chrome that repeats on every
page – navigation sidebars, header/footer menus, cookie banners. Such a file can be
>95% duplicated boilerplate. This drops every line (compared ignoring its trailing
newline) whose total occurrence count exceeds `min_repeats`, leaving the unique
content behind.

Structural markdown lines (blank lines, code fences, table separators) are protected
by `protect` so formatting survives no matter how often they recur. Pair this with
[`remove_improperly_double_newlines()`](#dn.repair.remove_improperly_double_newlines) afterwards to tidy the gaps left behind.

* **Parameters:**
  * **md** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The markdown text.
  * **min_repeats** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Lines occurring strictly more than this many times are removed. Defaults to 30.
  * **protect** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Predicate; lines for which it returns True are always kept. Defaults to
    protecting blank lines, code fences and table separators.
  * **keep_first** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, keep the first occurrence of each removed line (e.g. to retain one copy
    of a nav menu as a table of contents). Defaults to False (remove all occurrences,
    since boilerplate is pure noise in an AI context).
* **Returns:**
  The markdown with repeated boilerplate removed.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> md = "# Real title\nNAV\nunique A\nNAV\nunique B\nNAV\nunique C\nNAV\n"
>>> print(strip_repeated_lines(md, min_repeats=2))
# Real title
unique A
unique B
unique C

>>> print(strip_repeated_lines(md, min_repeats=2, keep_first=True))
# Real title
NAV
unique A
unique B
unique C
```
